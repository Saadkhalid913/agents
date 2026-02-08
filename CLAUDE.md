# Agents

A learning project for exploring AI agent paradigms through self-contained Python files. Each file implements a specific technique from first principles.

## File Naming

Agent files follow the pattern `N_descriptive_name.py` where N is a sequential integer (0, 1, 2, ...). Sub-scripts within a group use dotted prefixes (e.g., `1.1_`, `1.2_`).

Current files:
- `0_hello.py` — API connectivity test (OpenAI direct)
- `1_rag.py` — Interactive TUI orchestrator (questionary-based)
  - `1.1_in_context_qa.py` — In-context evaluation with LLM-as-judge
  - `1.2_naive_rag_with_embeddings.py` — Naive RAG with ChromaDB embeddings
  - `1.3_rag_with_query_rewording.py` — RAG with LLM-based query rewriting
  - `1.4_large_corpus_rag.ipynb` — RAG at scale with BEIR/NQ (10K+ docs, Recall@K)
  - `1.5_large_corpus_rag_with_query_rewriting.ipynb` — Query rewriting on large corpus
  - `1.6_reranking.ipynb` — Two-stage retrieval: broad embed search + LLM re-ranking
  - `1.7_hyde.ipynb` — HyDE: hypothetical document embeddings
  - `1.8_agentic_rag.ipynb` — Agentic RAG: retrieve-evaluate-reformulate loop
- `models.txt` — Available model IDs (one per line), used by `1_rag.py`

Scripts 1.1-1.3 accept CLI args (`--eval-model`, `--scoring-model`, `--num-examples`). Notebooks 1.4-1.8 use inline configuration cells. The orchestrator (`1_rag.py`) is interactive -- run it with `python 1_rag.py` and follow the prompts. Results are saved to `benchmark_results.json`.

## Code Style

- **Educational comments**: Each file is heavily commented explaining what and why
- **Docstrings**: Google-style with Args/Returns sections
- **Type hints**: Use throughout (list[str], Optional[int], tuple[int, str])
- **Self-contained**: Each file runs independently with `python N_name.py`
- **Constants at top**: Model names, configuration, and API setup at module level

## API Access

Two API providers are used:

| Provider   | Env Variable          | Used For                                |
|------------|-----------------------|-----------------------------------------|
| OpenAI     | `OPENAI_API_KEY`      | Direct API (hello, embeddings)          |
| OpenRouter | `OPENROUTER_API_KEY`  | Multi-model access (eval, scoring, RAG) |

Both use the OpenAI SDK. OpenRouter is accessed by setting `base_url="https://openrouter.ai/api/v1"`.

## Evaluation Pattern

Scripts 1.1-1.3 and notebooks 1.4-1.8 share a common evaluation framework:
1. Load dataset (HotpotQA for 1.1-1.3, BEIR/NQ for 1.4-1.8)
2. Generate answers with an eval model
3. Score answers with a stronger scoring model (gemini-3-flash)
4. Run in parallel via `ThreadPoolExecutor` (8 workers)
5. Print `RESULT_JSON:{...}` line for machine-readable output

## Documentation

Documentation lives in `docs/` as Markdown files that build into a single PDF.

```bash
./docs/build.sh          # Build PDF → docs/output/agents_report.pdf
./docs/build.sh --clean  # Remove generated PDF
```

Requires: pandoc, xelatex, Charter/Helvetica/DejaVu Sans Mono fonts.

Chapters follow the pattern `docs/NN_name.md`. Style is informal (Paul Graham blog style) with ASCII diagrams.

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create .env with OPENAI_API_KEY and OPENROUTER_API_KEY
```
