# Introduction

This is a project about building AI agents from scratch. Not the kind where you `pip install` a framework and hope for the best. The kind where you write every piece yourself and understand what's actually happening.

Each file is self-contained. You can read any one of them without reading the others. They're numbered in the order you should probably learn them, but there's no shared library or base class tying them together.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

Then run any script directly: `python 0_hello.py`.

## The files

```
0_hello.py                                API connectivity test
1_rag/
  1_rag.py                                Benchmark orchestrator (runs 1.1-1.3)
  1.1_in_context_qa.py                    In-context evaluation
  1.2_naive_rag_with_embeddings.py        Naive RAG with embeddings
  1.3_rag_with_query_rewording.py         RAG with query rewriting
  1.4_large_corpus_rag.ipynb              RAG at scale (10K+ docs)
  1.5_...query_rewriting.ipynb            Query rewriting at scale
  1.6_reranking.ipynb                     Two-stage retrieval with re-ranking
  1.7_hyde.ipynb                          Hypothetical document embeddings
  1.8_agentic_rag.ipynb                   Agentic RAG with feedback loop
2_tool_calls/
  2.1_single_tool_call.ipynb              Tool call lifecycle
  2.2_parallel_tool_calls.ipynb           Parallel tool execution
  2.3_chained_tool_calls.ipynb            Multi-turn chained tool calls
```
