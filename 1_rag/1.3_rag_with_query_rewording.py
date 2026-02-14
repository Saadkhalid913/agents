import os
import argparse
import openai
import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from datasets import load_dataset
from typing import Optional, Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables (including OPENROUTER_API_KEY)
load_dotenv()

# ============================================================================
# EVALUATION SETUP
# ============================================================================
# This script runs a RAG evaluation with QUERY RE-WORDING on HotpotQA.
#
# Improvement over naive RAG (script 2):
# Before embedding search, an LLM rewrites the user's question into a
# search-optimised query. This helps because:
# - User questions are conversational; embedding search works better with
#   keyword-rich, declarative statements
# - Multi-hop questions can be decomposed into the key facts being sought
# - Removing filler words and rephrasing improves cosine-similarity hits
#
# Pipeline per example:
#   1. Rewrite the question into an embedding-friendly search query (LLM call)
#   2. Embed documents into an ephemeral ChromaDB collection
#   3. Retrieve top-k documents using the REWRITTEN query
#   4. Pass retrieved docs + ORIGINAL question to the eval model
#   5. Score the answer with a stronger evaluator model
# ============================================================================

# Initialize OpenRouter client with the OpenAI SDK
client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Model being evaluated
EVAL_MODEL = "moonshotai/kimi-k2.5"

# Model used to rewrite queries — fast and cheap is fine here
REWRITE_MODEL = "google/gemini-3-flash-preview"

# Scoring model — stronger model for fair evaluation
SCORING_MODEL = "google/gemini-3-flash-preview"

# RAG retrieval settings
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"

# Embedding function for ChromaDB (uses OpenAI directly, not OpenRouter)
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL,
)

# Load the HotpotQA test dataset from Hugging Face
print("Loading HotpotQA dataset...")
ds = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")
print(f"Dataset loaded with {len(ds)} examples\n")


def rewrite_query(question: str) -> str:
    """
    Use an LLM to rewrite the user question into a search-optimised query
    that will better match relevant document embeddings.

    Args:
        question: The original user question

    Returns:
        A rewritten query string optimised for embedding retrieval
    """
    system_prompt = """You are a search query optimizer. Given a user question, rewrite it into a short, keyword-rich search query that will work well for semantic similarity search over a document collection.

Rules:
- Output ONLY the rewritten query, nothing else
- Remove filler words and conversational phrasing
- Keep all important entities, names, dates, and concepts
- For multi-hop questions, include the key facts being sought
- Keep it concise — 1-2 sentences max"""

    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=150,
    )

    rewritten = (response.choices[0].message.content or "").strip()
    # Fallback to original question if rewrite is empty
    return rewritten if rewritten else question


def generate_answer(question: str, documents: list[str], example_index: int = 0) -> tuple[str, str]:
    """
    Generate an answer using RAG with query re-wording:
    1. Rewrite the question for better retrieval
    2. Embed documents into a fresh ChromaDB collection
    3. Retrieve top-k docs using the rewritten query
    4. Pass retrieved docs + original question to the eval model

    Args:
        question: The original question to answer
        documents: List of document strings that serve as context
        example_index: Index to ensure unique collection names in parallel runs

    Returns:
        Tuple of (generated_answer, rewritten_query) so we can log both
    """
    # Step 1: Rewrite the query for better embedding retrieval
    rewritten_query = rewrite_query(question)

    # Step 2: Create a fresh ephemeral ChromaDB client per question
    chroma_client = chromadb.Client()
    collection_name = f"docs_{example_index}_{os.getpid()}"

    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=cast(Any, embedding_fn)
    )

    # Add all documents to the collection
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )

    # Step 3: Retrieve top-k documents using the REWRITTEN query
    results = collection.query(
        query_texts=[rewritten_query], n_results=min(TOP_K, len(documents)))

    # Safely extract retrieved documents
    retrieved_docs = []
    if results and results.get("documents"):
        docs_list = results["documents"]
        if docs_list and len(docs_list) > 0 and docs_list[0] is not None:
            retrieved_docs = docs_list[0]

    # Format retrieved documents into a readable context string
    context = "\n\n".join(
        [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)])

    # Step 4: Pass retrieved docs + ORIGINAL question to the eval model
    system_prompt = """You are a helpful assistant that answers questions based on provided documents.
Your task is to:
1. Carefully read all provided documents
2. Find the information needed to answer the question
3. Synthesize information across multiple documents if needed
4. Provide a clear, concise answer based ONLY on the documents

If the answer cannot be found in the documents, say so explicitly."""

    user_prompt = f"""Documents:
{context}

Question: {question}

Please answer the question based on the provided documents."""

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )

    # Clean up the collection
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    return response.choices[0].message.content or "", rewritten_query


def score_answer(
    question: str,
    documents: list[str],
    generated_answer: str,
    reference_answer: Optional[str] = None
) -> tuple[int, str]:
    """
    Score the generated answer using the evaluator model.

    Args:
        question: The original question
        documents: The context documents
        generated_answer: The answer generated by the eval model
        reference_answer: (Optional) A reference answer for comparison

    Returns:
        Tuple of (score out of 100, explanation)
    """
    context = "\n\n".join(
        [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)])

    system_prompt = """You are an expert evaluator assessing the quality of answers to questions.
Evaluate the answer on these criteria:
1. Correctness (0-25 points): Is the answer factually accurate based on the documents?
2. Completeness (0-25 points): Does it fully answer the question? Are important details included?
3. Faithfulness (0-25 points): Does it only use information from the documents? No hallucinations?
4. Clarity (0-25 points): Is the answer clear, well-organized, and easy to understand?

Respond with a JSON object containing:
{
    "score": <integer from 0-100>,
    "reasoning": "<brief explanation of the score>"
}"""

    eval_prompt = f"""Documents:
{context}

Question: {question}

Generated Answer:
{generated_answer}"""

    if reference_answer:
        eval_prompt += f"\n\nReference Answer (for context):\n{reference_answer}"

    response = client.chat.completions.create(
        model=SCORING_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": eval_prompt},
        ],
        max_tokens=300,
    )

    try:
        score_data = json.loads(response.choices[0].message.content or "{}")
        return score_data.get("score", 0), score_data.get("reasoning", "")
    except json.JSONDecodeError:
        return 0, "Error parsing score response"


def evaluate_single_example(example: dict, example_index: int) -> tuple[int, dict]:
    """
    Evaluate a single example — runs in parallel threads.

    Args:
        example: A single dataset example containing question, documents, response
        example_index: The index of this example in the dataset

    Returns:
        Tuple of (example_index, result_dict) for tracking and ordering
    """
    question = example["question"]
    documents = example["documents"]
    reference_answer = example.get("response", "")

    # Step 1: Generate answer (includes query rewriting internally)
    generated_answer, rewritten_query = generate_answer(
        question, documents, example_index)

    # Step 2: Score the answer
    score, reasoning = score_answer(
        question,
        documents,
        generated_answer,
        reference_answer
    )

    result = {
        "question": question,
        "rewritten_query": rewritten_query,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "score": score,
        "scoring_reasoning": reasoning,
    }

    return example_index, result


def run_evaluation(num_examples: Optional[int] = None, max_workers: int = 8) -> dict:
    """
    Run the complete evaluation on the dataset using parallel workers.

    Args:
        num_examples: If specified, only evaluate this many examples.
        max_workers: Number of parallel threads to use (default: 8)

    Returns:
        Dictionary with evaluation results and overall score
    """
    eval_size = num_examples if num_examples else len(ds)
    eval_size = min(eval_size, len(ds))

    print(
        f"Running evaluation on {eval_size} examples with {max_workers} parallel workers...\n")

    results_by_index = {}
    scores = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_single_example, ds[i], i): i
            for i in range(eval_size)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                example_idx, result = future.result()
                results_by_index[example_idx] = result
                scores.append(result["score"])
                completed += 1

                # Print progress with rewritten query visibility
                print(
                    f"[{completed}/{eval_size}] Completed example {example_idx + 1}")
                print(f"  Question:  {result['question'][:80]}...")
                print(f"  Rewritten: {result['rewritten_query'][:80]}...")
                print(f"  Score: {result['score']}/100")
                print(f"  Reasoning: {result['scoring_reasoning']}\n")

            except Exception as e:
                print(f"[Error] Example {idx + 1} failed: {e}\n")

    results = [results_by_index[i]
               for i in range(eval_size) if i in results_by_index]

    overall_score = sum(scores) / len(scores) if scores else 0

    evaluation_summary = {
        "model_evaluated": EVAL_MODEL,
        "rewrite_model": REWRITE_MODEL,
        "scoring_model": SCORING_MODEL,
        "num_examples_evaluated": len(scores),
        "overall_score": round(overall_score, 2),
        "individual_scores": scores,
        "score_distribution": {
            "90-100": sum(1 for s in scores if s >= 90),
            "80-89": sum(1 for s in scores if 80 <= s < 90),
            "70-79": sum(1 for s in scores if 70 <= s < 80),
            "60-69": sum(1 for s in scores if 60 <= s < 70),
            "below-60": sum(1 for s in scores if s < 60),
        },
        "detailed_results": results,
    }

    return evaluation_summary


if __name__ == "__main__":
    # ========================================================================
    # CLI ARGUMENTS
    # ========================================================================
    # When run standalone, uses defaults. When called from 1_rag.py, the
    # orchestrator passes --eval-model, --scoring-model, --rewrite-model,
    # and --num-examples to allow benchmarking different model combinations.
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="RAG evaluation with query re-wording on HotpotQA")
    parser.add_argument("--eval-model", default=EVAL_MODEL,
                        help="Model to evaluate (default: %(default)s)")
    parser.add_argument("--scoring-model", default=SCORING_MODEL,
                        help="Model for scoring answers (default: %(default)s)")
    parser.add_argument("--rewrite-model", default=REWRITE_MODEL,
                        help="Model for query rewriting (default: %(default)s)")
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of examples to evaluate (default: %(default)s)")
    args = parser.parse_args()

    # Override module-level constants with CLI args
    EVAL_MODEL = args.eval_model
    SCORING_MODEL = args.scoring_model
    REWRITE_MODEL = args.rewrite_model

    print("=" * 80)
    print("HOTPOTQA RAG EVALUATION WITH QUERY RE-WORDING")
    print("=" * 80 + "\n")

    results = run_evaluation(num_examples=args.num_examples)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model evaluated: {results['model_evaluated']}")
    print(f"Rewrite model:   {results['rewrite_model']}")
    print(f"Scoring model:   {results['scoring_model']}")
    print(f"Examples evaluated: {results['num_examples_evaluated']}")
    print(f"\nOverall Score: {results['overall_score']}/100")
    print(f"\nScore Distribution:")
    for range_label, count in results['score_distribution'].items():
        print(f"  {range_label}: {count} examples")

    # Machine-readable result line for orchestrator (1_rag.py)
    result_json = {
        "overall_score": results["overall_score"],
        "num_examples": results["num_examples_evaluated"],
        "score_distribution": results["score_distribution"],
    }
    print(f"RESULT_JSON:{json.dumps(result_json)}")
