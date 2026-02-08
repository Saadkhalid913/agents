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
0_hello.py                       API connectivity test
1_rag.py                         Benchmark orchestrator (runs 1.1-1.3)
1.1_in_context_qa.py             In-context evaluation
1.2_naive_rag_with_embeddings.py Naive RAG with embeddings
1.3_rag_with_query_rewording.py  RAG with query rewriting
1.4_large_corpus_rag.py          RAG at scale (10K+ docs)
```
