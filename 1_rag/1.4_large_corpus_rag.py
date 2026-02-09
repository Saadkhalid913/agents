import os
import argparse
import random
import time
import openai
import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from datasets import load_dataset
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables (including OPENROUTER_API_KEY and OPENAI_API_KEY)
load_dotenv()

# ============================================================================
# LARGE CORPUS RAG EVALUATION
# ============================================================================
# Previous scripts (1.1-1.3) used ~10 documents per question. That barely
# tests retrieval — when the haystack has 10 items, even bad embeddings
# find the needle.
#
# This script uses the BEIR Natural Questions dataset: 2.68M Wikipedia
# passages and 3,452 Google Search questions with ground-truth relevance
# labels. We build ONE shared ChromaDB collection with thousands of
# documents and measure both retrieval quality and answer quality.
#
# Key differences from 1.2 (naive RAG):
#   - ONE shared collection vs. per-question ephemeral collections
#   - 10,000+ documents vs. ~10 documents
#   - Ground-truth relevance labels to measure Recall@K
#   - Persistent ChromaDB so we don't re-embed on every run
#   - BEIR/NQ dataset instead of RAGBench/HotpotQA
#
# The corpus is built by taking all gold-relevant documents for our eval
# queries, then padding with randomly sampled Wikipedia passages as
# distractors. This guarantees the needle IS in the haystack.
# ============================================================================

# Initialize OpenRouter client with the OpenAI SDK
client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Model being evaluated
EVAL_MODEL = "moonshotai/kimi-k2.5"

# Scoring model
SCORING_MODEL = "google/gemini-3-flash-preview"

# RAG retrieval settings
TOP_K = 5
CORPUS_SIZE = 10000
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = ".chroma_nq"

# Embedding function for ChromaDB (uses OpenAI directly, not OpenRouter)
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL,
)


def load_data(corpus_size: int, num_examples: int) -> tuple[list[dict], list[dict], dict]:
    """
    Load BEIR Natural Questions: corpus, queries, and relevance judgments.

    Builds a corpus subset that includes all gold-relevant documents for
    the queries we'll evaluate, plus randomly sampled Wikipedia passages
    as distractors up to corpus_size.

    Args:
        corpus_size: Total number of documents in the corpus subset
        num_examples: Number of queries to evaluate

    Returns:
        Tuple of (corpus_subset, eval_queries, qrels)
        - corpus_subset: list of dicts with _id, title, text
        - eval_queries: list of dicts with _id, text
        - qrels: dict mapping query_id -> set of relevant doc_ids
    """
    print("Loading BEIR/NQ queries and relevance judgments...")
    # The BeIR/nq dataset uses a custom loading script that datasets v4+
    # no longer supports. Load directly from the auto-converted parquet files.
    queries_ds = load_dataset(
        "parquet",
        data_files="hf://datasets/BeIR/nq@refs/convert/parquet/queries/queries/0000.parquet",
        split="train",
    )
    qrels_ds = load_dataset("BeIR/nq-qrels", split="test")

    # Build qrels lookup: query_id -> set of relevant corpus doc_ids
    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

    # Select queries that have relevance judgments
    eval_queries = [q for q in queries_ds if q["_id"] in qrels][:num_examples]

    # Collect all gold-relevant doc IDs for the queries we'll evaluate
    gold_doc_ids: set[str] = set()
    for q in eval_queries:
        gold_doc_ids.update(qrels[q["_id"]])

    print(
        f"Selected {len(eval_queries)} queries, {len(gold_doc_ids)} gold documents")

    # Load the full corpus (HuggingFace uses memory-mapped Arrow, so this
    # doesn't load 2.68M docs into RAM — it's a lazy view)
    print("Loading BEIR/NQ corpus (2.68M Wikipedia passages)...")
    corpus_ds = load_dataset(
        "parquet",
        data_files=[
            "hf://datasets/BeIR/nq@refs/convert/parquet/corpus/corpus/0000.parquet",
            "hf://datasets/BeIR/nq@refs/convert/parquet/corpus/corpus/0001.parquet",
            "hf://datasets/BeIR/nq@refs/convert/parquet/corpus/corpus/0002.parquet",
        ],
        split="train",
    )

    # Find gold documents using HF's optimized batched filter
    print("Locating gold documents in corpus...")
    gold_docs_ds = corpus_ds.filter(
        lambda batch: [did in gold_doc_ids for did in batch["_id"]],
        batched=True,
        batch_size=10000,
    )
    print(f"Found {len(gold_docs_ds)}/{len(gold_doc_ids)} gold documents")

    # Build corpus subset: gold docs + random distractors
    # Skip docs with empty text (OpenAI embeddings API rejects them)
    corpus_subset = [
        {"_id": gold_docs_ds[i]["_id"],
         "text": gold_docs_ds[i]["text"],
         "title": gold_docs_ds[i]["title"]}
        for i in range(len(gold_docs_ds))
        if gold_docs_ds[i]["text"]
    ]

    # Sample random distractor documents from the corpus
    fill_count = max(0, corpus_size - len(corpus_subset))
    if fill_count > 0:
        random.seed(42)
        # Sample candidate indices, then filter out any gold docs
        candidate_indices = random.sample(
            range(len(corpus_ds)), min(fill_count * 2, len(corpus_ds)))
        candidates = corpus_ds.select(candidate_indices)

        for i in range(len(candidates)):
            if candidates[i]["_id"] not in gold_doc_ids and candidates[i]["text"]:
                corpus_subset.append({
                    "_id": candidates[i]["_id"],
                    "text": candidates[i]["text"],
                    "title": candidates[i]["title"],
                })
            if len(corpus_subset) >= corpus_size:
                break

    gold_count = len(gold_docs_ds)
    fill_actual = len(corpus_subset) - gold_count
    print(f"Corpus: {len(corpus_subset)} docs "
          f"({gold_count} gold + {fill_actual} distractors)\n")

    return corpus_subset, eval_queries, qrels


def build_collection(
    corpus_subset: list[dict],
    chroma_dir: str,
) -> chromadb.Collection:
    """
    Build or load a persistent ChromaDB collection from the corpus subset.

    Uses a stable collection name based on a hash of the document IDs, so
    re-running with the same corpus skips embedding entirely (even across
    sessions). If the corpus changes, a new collection is created.

    Args:
        corpus_subset: List of document dicts with _id, text, title
        chroma_dir: Path to ChromaDB persistent storage directory

    Returns:
        A ChromaDB Collection ready for querying
    """
    import hashlib

    # Build a stable name from the sorted doc IDs so the same corpus
    # always maps to the same collection (and we skip re-embedding)
    id_hash = hashlib.sha256(
        ",".join(sorted(d["_id"] for d in corpus_subset)).encode()
    ).hexdigest()[:12]
    collection_name = f"nq_{len(corpus_subset)}_{id_hash}"

    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Try to reuse an existing collection with matching name and size
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=cast(Any, embedding_fn),
        )
        if collection.count() == len(corpus_subset):
            print(f"Cache hit! Reusing collection '{collection_name}' "
                  f"({collection.count():,} docs, no re-embedding needed)\n")
            return collection
        # Size mismatch -- rebuild
        print(f"Collection size mismatch "
              f"({collection.count()} vs {len(corpus_subset)}), rebuilding...")
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    total = len(corpus_subset)
    print(
        f"Embedding {total:,} documents into collection '{collection_name}'...")
    print(f"(This is a one-time cost -- cached on disk for future runs)\n")

    # Embed and add documents in batches with timing stats
    BATCH_SIZE = 500
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=cast(Any, embedding_fn),
    )

    start_time = time.time()
    for batch_num, i in enumerate(range(0, total, BATCH_SIZE), 1):
        batch = corpus_subset[i:i + BATCH_SIZE]
        batch_start = time.time()

        collection.add(
            ids=[d["_id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"title": d["title"]} for d in batch],
        )

        done = min(i + BATCH_SIZE, total)
        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        rate = done / elapsed  # docs per second
        remaining = (total - done) / rate if rate > 0 else 0

        print(f"  [{batch_num}/{total_batches}] {done:,}/{total:,} docs | "
              f"batch: {batch_time:.1f}s | "
              f"rate: {rate:.0f} docs/s | "
              f"ETA: {remaining:.0f}s")

    total_time = time.time() - start_time
    print(f"\nCollection ready: {collection.count():,} documents "
          f"(embedded in {total_time:.1f}s)\n")
    return collection


def generate_answer(question: str, retrieved_docs: list[str]) -> str:
    """
    Generate an answer using the eval model with retrieved documents.

    Args:
        question: The question to answer
        retrieved_docs: List of document texts retrieved from the collection

    Returns:
        The generated answer as a string
    """
    context = "\n\n".join(
        [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)])

    system_prompt = """You are a helpful assistant that answers questions based on provided documents.
Your task is to:
1. Carefully read all provided documents
2. Find the information needed to answer the question
3. Provide a clear, concise answer based ONLY on the documents

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

    return response.choices[0].message.content or ""


def score_answer(
    question: str,
    documents: list[str],
    generated_answer: str,
) -> tuple[int, str]:
    """
    Score the generated answer using the scoring model.

    Args:
        question: The original question
        documents: The retrieved documents used to generate the answer
        generated_answer: The answer generated by the eval model

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


def evaluate_single_example(
    query: dict,
    example_index: int,
    collection: chromadb.Collection,
    qrels: dict[str, set[str]],
    top_k: int,
) -> tuple[int, dict]:
    """
    Evaluate a single query against the shared collection.

    Retrieves top-k documents, computes Recall@K against ground-truth
    relevance labels, generates an answer, and scores it.

    Args:
        query: Dict with _id and text fields
        example_index: Index for ordering results
        collection: The shared ChromaDB collection to search
        qrels: Ground-truth relevance labels (query_id -> set of doc_ids)
        top_k: Number of documents to retrieve

    Returns:
        Tuple of (example_index, result_dict)
    """
    question = query["text"]
    query_id = query["_id"]
    gold_ids = qrels.get(query_id, set())

    # Retrieve from the shared collection
    results = collection.query(query_texts=[question], n_results=top_k)

    retrieved_ids = results["ids"][0] if results["ids"] else []
    retrieved_docs = results["documents"][0] if results["documents"] else []

    # Compute Recall@K: what fraction of gold docs did we find?
    hits = len(gold_ids & set(retrieved_ids))
    recall = hits / len(gold_ids) if gold_ids else 0.0

    # Generate and score
    generated_answer = generate_answer(question, retrieved_docs)
    score, reasoning = score_answer(question, retrieved_docs, generated_answer)

    result = {
        "query_id": query_id,
        "question": question,
        "generated_answer": generated_answer,
        "score": score,
        "scoring_reasoning": reasoning,
        "recall_at_k": recall,
        "gold_docs_found": hits,
        "gold_docs_total": len(gold_ids),
    }

    return example_index, result


def run_evaluation(
    eval_queries: list[dict],
    collection: chromadb.Collection,
    qrels: dict[str, set[str]],
    top_k: int,
    max_workers: int = 8,
) -> dict:
    """
    Run evaluation on all queries using parallel workers.

    Args:
        eval_queries: List of query dicts to evaluate
        collection: The shared ChromaDB collection
        qrels: Ground-truth relevance labels
        top_k: Number of documents to retrieve per query
        max_workers: Number of parallel threads (default: 8)

    Returns:
        Dictionary with evaluation results, scores, and retrieval metrics
    """
    eval_size = len(eval_queries)

    print(f"Running evaluation on {eval_size} queries with "
          f"{max_workers} parallel workers...\n")

    results_by_index: dict[int, dict] = {}
    scores: list[int] = []
    recalls: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_single_example,
                eval_queries[i], i, collection, qrels, top_k
            ): i
            for i in range(eval_size)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                example_idx, result = future.result()
                results_by_index[example_idx] = result
                scores.append(result["score"])
                recalls.append(result["recall_at_k"])
                completed += 1

                # Print progress
                print(
                    f"[{completed}/{eval_size}] Example {example_idx + 1}")
                print(f"  Question: {result['question'][:80]}...")
                print(f"  Recall@{top_k}: {result['recall_at_k']:.2f} "
                      f"({result['gold_docs_found']}/{result['gold_docs_total']} gold)")
                print(f"  Score: {result['score']}/100")
                print(f"  Reasoning: {result['scoring_reasoning']}\n")

            except Exception as e:
                print(f"[Error] Example {idx + 1} failed: {e}\n")

    results = [results_by_index[i]
               for i in range(eval_size) if i in results_by_index]

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    return {
        "model_evaluated": EVAL_MODEL,
        "scoring_model": SCORING_MODEL,
        "dataset": "BeIR/NQ",
        "corpus_size": collection.count(),
        "top_k": top_k,
        "num_examples_evaluated": len(scores),
        "overall_score": round(avg_score, 2),
        "avg_recall_at_k": round(avg_recall, 4),
        "individual_scores": scores,
        "individual_recalls": recalls,
        "score_distribution": {
            "90-100": sum(1 for s in scores if s >= 90),
            "80-89": sum(1 for s in scores if 80 <= s < 90),
            "70-79": sum(1 for s in scores if 70 <= s < 80),
            "60-69": sum(1 for s in scores if 60 <= s < 70),
            "below-60": sum(1 for s in scores if s < 60),
        },
        "detailed_results": results,
    }


if __name__ == "__main__":
    # ========================================================================
    # CLI ARGUMENTS
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Large corpus RAG evaluation on BEIR/NQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python 1.4_large_corpus_rag.py                         # defaults (10K docs)
  python 1.4_large_corpus_rag.py --corpus-size 50000     # bigger haystack
  python 1.4_large_corpus_rag.py --num-examples 10       # quick test
  python 1.4_large_corpus_rag.py --top-k 10              # retrieve more docs
""")
    parser.add_argument("--eval-model", default=EVAL_MODEL,
                        help="Model to evaluate (default: %(default)s)")
    parser.add_argument("--scoring-model", default=SCORING_MODEL,
                        help="Model for scoring answers (default: %(default)s)")
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of queries to evaluate (default: %(default)s)")
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE,
                        help="Number of documents in corpus (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Number of documents to retrieve per query (default: %(default)s)")
    args = parser.parse_args()

    # Override module-level constants with CLI args
    EVAL_MODEL = args.eval_model
    SCORING_MODEL = args.scoring_model

    print("=" * 80)
    print("BEIR/NQ LARGE CORPUS RAG EVALUATION")
    print("=" * 80)
    print(f"Eval model:    {EVAL_MODEL}")
    print(f"Scoring model: {SCORING_MODEL}")
    print(f"Corpus size:   {args.corpus_size:,} documents")
    print(f"Top-K:         {args.top_k}")
    print(f"Num queries:   {args.num_examples}")
    print("=" * 80 + "\n")

    # Step 1: Load data and build corpus subset
    corpus_subset, eval_queries, qrels = load_data(
        args.corpus_size, args.num_examples)

    # Step 2: Build or load the ChromaDB collection
    collection = build_collection(corpus_subset, CHROMA_DIR)

    # Step 3: Run evaluation
    results = run_evaluation(
        eval_queries, collection, qrels, args.top_k)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model evaluated:  {results['model_evaluated']}")
    print(f"Scoring model:    {results['scoring_model']}")
    print(f"Dataset:          {results['dataset']}")
    print(f"Corpus size:      {results['corpus_size']:,} documents")
    print(f"Top-K:            {results['top_k']}")
    print(f"Queries evaluated: {results['num_examples_evaluated']}")
    print(f"\nAvg Recall@{args.top_k}:    {results['avg_recall_at_k']:.4f}")
    print(f"Overall Score:    {results['overall_score']}/100")
    print(f"\nScore Distribution:")
    for range_label, count in results['score_distribution'].items():
        print(f"  {range_label}: {count} examples")

    # Save results to ./evals
    evals_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    eval_path = os.path.join(evals_dir, "1.4_large_corpus_rag.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")

    # Machine-readable result line for orchestrator
    result_json = {
        "overall_score": results["overall_score"],
        "avg_recall_at_k": results["avg_recall_at_k"],
        "num_examples": results["num_examples_evaluated"],
        "corpus_size": results["corpus_size"],
        "score_distribution": results["score_distribution"],
    }
    print(f"RESULT_JSON:{json.dumps(result_json)}")
