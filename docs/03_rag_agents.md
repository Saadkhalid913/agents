# RAG

The interesting question isn't whether LLMs can answer questions from documents. They can. The interesting question is what happens when you stop handing them the documents and make them go find the right ones.

That's retrieval-augmented generation: embed documents into vectors, search for the ones that match the question, then feed only those to the model. The rest of this chapter is about what happens when you try it, what breaks, and how to fix it.

## The Orchestrator (`1_rag/1_rag.py`)

Before diving into individual scripts, it's worth mentioning the benchmark runner. It takes model parameters once and runs all the RAG approaches back-to-back:

```bash
python 1_rag/1_rag.py --eval-model openai/gpt-4o --num-examples 20
```

This produces a comparison table at the end so you can see how the approaches stack up against each other with identical settings.

## Naive RAG (`1_rag/1.2_naive_rag_with_embeddings.py`)

The simplest possible RAG pipeline: embed the documents, embed the question, find the top 3 matches, and feed them to the model.

```text
   question + all documents
              │
   ┌──────────▼───────────┐
   │  Embed all docs into  │     OpenAI
   │  ChromaDB collection  │ <── text-embedding-3-small
   └──────────┬────────────┘
              │
   ┌──────────▼───────────┐
   │  Query: find top 3    │
   │  most similar docs    │
   └──────────┬────────────┘
              │
       top 3 documents
              │
   ┌──────────▼───────────┐
   │  Eval model           │
   │  "Answer using these" │
   └──────────┬────────────┘
              │
       generated answer ──> score it
```

The crucial difference from 1.1: the model no longer sees all the documents. It sees *only* what the retrieval found. If the embedding search misses a relevant document, the model can't use it.

Each question gets its own ephemeral ChromaDB collection with only its ~10 documents. This is "naive" because in real RAG systems you don't rebuild the index per question --- you search a shared corpus. But for measuring the impact of retrieval vs. in-context, this controlled setup is useful.

Run it:

```bash
python 1_rag/1.2_naive_rag_with_embeddings.py
python 1_rag/1.2_naive_rag_with_embeddings.py --eval-model openai/gpt-4o
```

## Query Rewriting (`1_rag/1.3_rag_with_query_rewording.py`)

Here's an insight that's easy to miss: the question the user asks isn't always the best query for embedding search.

User questions are conversational. Embedding search works better with keyword-rich, declarative statements. So before searching, we ask an LLM to rewrite the question into a search-optimized query.

```text
   "Who was the director of the 1993 film that featured
    a song by Aerosmith?"
              │
   ┌──────────▼───────────┐
   │  Rewrite model        │
   │  (gemini-3-flash)     │
   └──────────┬────────────┘
              │
   "1993 film director Aerosmith song soundtrack"
              │
   ┌──────────▼───────────┐
   │  Embed + retrieve     │
   │  top 3 docs           │
   └──────────┬────────────┘
              │
       top 3 documents + ORIGINAL question
              │
   ┌──────────▼───────────┐
   │  Eval model answers   │
   └──────────────────────┘
```

Notice the subtle but important detail: we search with the *rewritten* query but answer with the *original* question. The rewrite is optimized for retrieval, not for comprehension.

Everything else is identical to 1.2. Same scoring, same dataset, same parallelism. The only change is one extra LLM call per question to rewrite the query before searching.

Run it:

```bash
python 1_rag/1.3_rag_with_query_rewording.py
python 1_rag/1.3_rag_with_query_rewording.py --rewrite-model openai/gpt-4o-mini
```

## Scaling Up (`1_rag/1.4_large_corpus_rag.ipynb`)

Everything so far has a problem: the haystack is tiny. Each question in HotpotQA comes with about 10 documents. Finding the right one out of 10 isn't much of a test.

Real RAG systems search thousands or millions of documents. This notebook uses the BEIR Natural Questions dataset --- 2.68 million Wikipedia passages and 3,452 questions originally from Google Search. We build one shared ChromaDB collection with 10,000+ documents and make every query search the same big corpus.

```text
   BEIR/NQ: 2.68M Wikipedia passages
              │
   ┌──────────▼───────────┐
   │  Sample 10,000 docs   │     (gold docs + random distractors)
   │  Build shared index   │ <── embed once, reuse across queries
   └──────────┬────────────┘
              │
   for each query:
   ┌──────────▼───────────┐
   │  Search the shared    │
   │  10K-doc collection   │
   └──────────┬────────────┘
              │
       top 5 documents
              │
   ┌──────────▼───────────┐
   │  Generate + score     │
   └──────────┬────────────┘
              │
   Recall@K + answer score
```

The key architectural difference: one persistent collection shared across all queries, instead of throwaway per-question collections. The collection is cached on disk (ChromaDB PersistentClient) so you don't re-embed on every run.

We also introduce a new metric: **Recall@K**. For each query, the BEIR dataset tells us which documents are actually relevant (ground-truth labels). Recall@K measures what fraction of those we found in our top-K retrieval. If the gold document is ranked 6th and K=5, recall is 0.

This is where you start to see the real challenge. When the haystack grows from 10 to 10,000 documents, retrieval quality drops and it drags answer quality down with it.

## Query Rewriting at Scale (`1_rag/1.5_large_corpus_rag_with_query_rewriting.ipynb`)

1.3 showed that query rewriting helps on small datasets. 1.5 tests whether the same trick works when the haystack is 10,000 documents instead of 10.

The approach is identical to 1.4, with one extra step: before searching, we rewrite the question into a search-optimized query. The intuition is the same as 1.3, but now we're testing it where it matters --- at scale, where retrieval is actually hard.

```text
   "Who was the director of the 1993 film
    that featured a song by Aerosmith?"
              │
   ┌──────────▼───────────┐
   │  Rewrite model        │     gemini-3-flash
   │  "optimize for        │
   │   embedding search"   │
   └──────────┬────────────┘
              │
   "1993 film director Aerosmith song soundtrack"
              │
   ┌──────────▼───────────┐
   │  Search 10K-doc       │
   │  ChromaDB collection  │
   └──────────┬────────────┘
              │
       top 5 documents + ORIGINAL question
              │
   ┌──────────▼───────────┐
   │  Generate + score     │
   └──────────────────────┘
```

The core rewriting logic:

```python
def rewrite_query(question: str) -> str:
    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a search query optimizer. Rewrite the "
                "question as a keyword-rich, declarative query "
                "optimized for semantic similarity search. "
                "Remove filler words. Keep it concise. "
                "Output ONLY the rewritten query."
            )},
            {"role": "user", "content": question},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
```

The subtle detail: we search with the *rewritten* query but answer with the *original* question. The rewrite is optimized for retrieval, not comprehension.

## Re-ranking (`1_rag/1.6_reranking.ipynb`)

Embedding similarity is fast but shallow. It matches on surface-level vector distance, which means a document about "apple pie recipes" might rank higher than one about "Apple Inc. revenue" for a query about Apple's stock price, just because the word "apple" is prominent.

Re-ranking fixes this by adding a second stage: retrieve broadly with embeddings, then have an LLM actually *read* each candidate and score its relevance.

```text
   question
      │
   ┌──▼───────────────────┐
   │  Stage 1: Embeddings  │     fast, shallow
   │  Retrieve top 50      │
   └──┬────────────────────┘
      │
   50 candidate documents
      │
   ┌──▼───────────────────┐
   │  Stage 2: LLM         │     slow, precise
   │  Score each doc 0-10   │
   │  Keep top 5            │
   └──┬────────────────────┘
      │
   top 5 re-ranked documents
      │
   ┌──▼───────────────────┐
   │  Generate + score     │
   └──────────────────────┘
```

The key trade-off: INITIAL_K=50 gives you broad coverage (you're unlikely to miss the right document), while RERANK_K=5 gives the answer model a focused, high-quality context. You're spending one LLM call per query to read 50 document snippets, but that's much cheaper than embedding-searching the entire corpus at higher precision.

The re-ranking function asks the model to score each document's relevance:

```python
def rerank_documents(question, doc_ids, doc_texts, top_k):
    doc_list = "\n\n".join(
        f"[Doc {i+1}] {text[:500]}"
        for i, text in enumerate(doc_texts)
    )
    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{
            "role": "system",
            "content": (
                "Score each document's relevance to the "
                "question on a 0-10 scale. Respond with "
                "ONLY a JSON array: "
                '[{"doc": 1, "score": 8}, ...]'
            )
        }, {
            "role": "user",
            "content": f"Question: {question}\n\n"
                       f"Documents:\n{doc_list}"
        }],
        max_tokens=1000,
    )
    # Parse scores, sort descending, keep top_k
    scores = json.loads(response.choices[0].message.content)
    scored = sorted(
        [(s["score"], s["doc"] - 1) for s in scores],
        reverse=True,
    )
    top_indices = [idx for _, idx in scored[:top_k]]
    return (
        [doc_ids[i] for i in top_indices],
        [doc_texts[i] for i in top_indices],
    )
```

The evaluation tracks both `recall_before_rerank` (what embedding search found in the top 5) and `recall_at_k` (what's left after re-ranking). This tells you whether the LLM is helping or hurting --- it's possible for re-ranking to *drop* a relevant document if the model misjudges its relevance.

## HyDE --- Hypothetical Document Embeddings (`1_rag/1.7_hyde.ipynb`)

HyDE flips the retrieval problem. Instead of embedding the question and searching for similar documents, you generate a *hypothetical answer* and search for documents similar to *that*.

Why does this work? Consider the embedding space. A question like "what is the capital of France?" and a Wikipedia passage containing "Paris is the capital of France" live in different neighborhoods --- one is a question, the other is a statement. But a hypothetical answer ("The capital of France is Paris, located in the Ile-de-France region") is much closer to the real document in embedding space.

The hypothetical answer doesn't need to be correct. It just needs to *sound like* the kind of document that would contain the real answer.

```text
   "What is the capital of France?"
              │
   ┌──────────▼───────────┐
   │  HyDE model           │     gemini-3-flash
   │  Generate hypothetical │
   │  Wikipedia passage     │
   └──────────┬────────────┘
              │
   "Paris is the capital of France.
    It is located in the
    Ile-de-France region..."
              │
   ┌──────────▼───────────┐
   │  Embed the HYPOTHETICAL│     OpenAI
   │  document (not the     │     text-embedding-3-small
   │  question!)            │
   └──────────┬────────────┘
              │
   ┌──────────▼───────────┐
   │  Search for similar    │
   │  REAL documents        │
   └──────────┬────────────┘
              │
       top 5 real documents + ORIGINAL question
              │
   ┌──────────▼───────────┐
   │  Generate + score     │
   └──────────────────────┘
```

The implementation has two key functions:

```python
def generate_hypothetical_document(question: str) -> str:
    """Generate a fake Wikipedia passage that would answer
    the question. Doesn't need to be correct -- just needs
    to sound like a real article about the topic."""
    response = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[{
            "role": "system",
            "content": "Write a short Wikipedia-style passage "
                       "that would answer the question. Keep "
                       "it under 150 words. Output ONLY the "
                       "passage."
        }, {"role": "user", "content": question}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def hyde_retrieve(question, collection, top_k):
    hypo_doc = generate_hypothetical_document(question)
    # Embed the hypothetical doc, not the question
    hypo_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[hypo_doc]
    ).data[0].embedding
    # Search using the hypothetical doc's embedding
    results = collection.query(
        query_embeddings=[hypo_embedding],
        n_results=top_k,
    )
    return results["ids"][0], results["documents"][0], hypo_doc
```

Notice HyDE requires a direct OpenAI client for embedding (not through ChromaDB), since we're embedding arbitrary text rather than using ChromaDB's built-in query.

## Agentic RAG (`1_rag/1.8_agentic_rag.ipynb`)

Every technique so far uses a fixed pipeline: retrieve once, answer once. If the retrieval misses, you're stuck.

Agentic RAG adds a feedback loop. After retrieving documents, an LLM evaluates whether they actually contain enough information to answer the question. If not, it reformulates the query and tries again. This is the first technique in our progression where the model makes its own decisions about the retrieval process.

```text
   question
      │
      ▼
   ┌──┴───────────────────┐
   │  Retrieve top 5       │<────────┐
   └──┬────────────────────┘         │
      │                              │
   ┌──▼───────────────────┐         │
   │  Agent evaluates:     │         │
   │  "Are these docs      │   reformulated
   │   sufficient?"        │     query
   └──┬────────┬───────────┘         │
      │        │                     │
     YES      NO                     │
      │        │                     │
      │   ┌────▼──────────────┐     │
      │   │  Reformulate query │─────┘
      │   │  (different angle, │
      │   │   new keywords)    │  up to MAX_RETRIES
      │   └────────────────────┘
      │
   ┌──▼───────────────────┐
   │  Generate + score     │
   └──────────────────────┘
```

The agent's evaluation step is the key innovation:

```python
def evaluate_retrieval(question, doc_texts):
    """Ask the LLM: do these docs contain enough
    info to answer the question?"""
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{
            "role": "system",
            "content": "Decide if the documents contain enough "
                       "information to answer the question. "
                       'Respond with JSON: {"sufficient": '
                       'true/false, "reformulated_query": '
                       '"new query if not sufficient"}'
        }, {
            "role": "user",
            "content": f"Question: {question}\n\n"
                       f"Documents:\n{doc_list}"
        }],
        max_tokens=200,
    )
    data = json.loads(response.choices[0].message.content)
    return data["sufficient"], data["reformulated_query"]
```

The retrieval loop merges results across attempts (deduplicating by document ID), so each retry *adds* to the pool of candidates rather than replacing them. After MAX_RETRIES=3 attempts, it answers with the best documents it has.

This trades latency for quality. If the first retrieval hits, you get an answer in one round. If it misses, you spend 2--3 extra LLM calls trying different angles. The evaluation tracks `retrieval_attempts` and `queries_tried` per example so you can see how often the agent actually retries.

## What we learned

The progression from 1.1 to 1.8 tells a clear story about the retrieval problem and the different ways to attack it.

**1.1 (all documents)** is the ceiling. The model has everything it needs. If it can't answer, the problem is comprehension, not retrieval.

**1.2 (naive RAG)** shows that retrieval works surprisingly well when the haystack is small. With only 10 documents, even simple embedding search finds what you need most of the time.

**1.3 (query rewriting)** demonstrates that the question you ask isn't always the best search query. A quick rewrite step is cheap and often improves retrieval.

**1.4 (large corpus)** is where reality sets in. When the haystack grows to 10,000+ documents, naive top-K retrieval starts missing relevant content, and answer quality drops.

**1.5 (query rewriting at scale)** tests whether the trick from 1.3 still helps when the corpus is large. Same technique, harder problem.

**1.6 (re-ranking)** attacks the problem from the other end: retrieve broadly, then use an LLM to filter precisely. The cost is one extra LLM call per query to score 50 documents.

**1.7 (HyDE)** changes *what* gets embedded for the search. Instead of embedding the question, embed a hypothetical answer. This bridges the question-document gap in embedding space.

**1.8 (agentic RAG)** gives the model autonomy over the retrieval process. Instead of a fixed pipeline, it decides whether to try again and how to reformulate. This is the most expensive approach but the most adaptive.

Each technique adds one idea. They can also be combined --- you could rewrite the query (1.5), retrieve broadly (1.6 stage 1), re-rank (1.6 stage 2), and retry if the agent isn't satisfied (1.8). Production RAG systems often stack several of these together.
