# Agents

A from-scratch exploration of AI agent techniques. Every script implements one idea — no frameworks, no abstractions, just the raw patterns that make agents work.

The project is structured as a series of self-contained experiments, numbered in the order you should learn them. Each one builds on concepts from the last, but you can read any file on its own. The accompanying [PDF report](docs/) explains the theory and results.

## What this covers

The goal is exhaustive coverage of agent paradigms, from the most basic retrieval pipeline to state-of-the-art multi-agent architectures. Each section lives in its own folder.

| # | Section | Status | What you'll learn |
|---|---------|--------|-------------------|
| 0 | [Hello](0_hello.py) | Done | API connectivity, verify the plumbing works |
| 1 | [RAG](1_rag/) | Done | Retrieval-augmented generation: naive RAG, query rewriting, re-ranking, HyDE, agentic RAG. Small-scale (HotpotQA) and large-scale (BEIR/NQ, 10K+ docs). Evaluation with LLM-as-judge. |
| 2 | [Tool Calling](2_tool_calls/) | Planning | Function calling, parallel/chained tool use, error handling, ReAct loop, tool selection at scale, structured outputs, MCP |
| 3 | Planning & Reasoning | Planned | Chain-of-thought, tree-of-thought, self-consistency, plan-and-execute, reflection |
| 4 | Code Generation | Planned | Sandboxed execution, generate-test-debug loops, code agents |
| 5 | Memory | Planned | Conversation memory, vector long-term memory, episodic memory, memory-augmented agents |
| 6 | Multi-Agent Systems | Planned | Agent debate, hierarchical delegation, role-based crews, handoff protocols |
| 7 | Autonomous Agents | Planned | Goal decomposition, task prioritization, skill libraries, budget-constrained loops |
| 8 | Evaluation & Safety | Planned | Hallucination detection, guardrails, agent trajectory scoring, benchmarks |
| 9 | Infrastructure | Planned | MCP servers/clients, agent communication protocols, tool registries |
| 10 | Long-Running Agents | Planned | Checkpointing, human-in-the-loop, async task queues, approval workflows |
| 11 | Multimodal Agents | Planned | Vision + tools, document QA, screenshot-based agents |
| 12 | Fine-Tuning for Agency | Planned | LoRA for tool use, trajectory distillation, reward models for agent behavior |

### Section 1: RAG (complete)

Eight experiments that tell the story of retrieval:

```
1_rag/
  1_rag.py                          TUI orchestrator (runs benchmarks back-to-back)
  1.1_in_context_qa.py              Baseline: give the model everything, measure the ceiling
  1.2_naive_rag_with_embeddings.py  Simplest RAG: embed, search, answer
  1.3_rag_with_query_rewording.py   Rewrite the question before searching
  1.4_large_corpus_rag.ipynb        Scale to 10K+ docs (BEIR/NQ dataset)
  1.5_...query_rewriting.ipynb      Query rewriting at scale
  1.6_reranking.ipynb               Broad retrieval + LLM re-ranking
  1.7_hyde.ipynb                    Search with hypothetical answers
  1.8_agentic_rag.ipynb             Feedback loop: retrieve, evaluate, reformulate
```

### Section 2: Tool Calling (in progress)

Planned experiments covering the mechanics of how LLMs call functions:

- Single tool calls (the lifecycle)
- Parallel and chained tool calls
- Error handling and recovery
- The ReAct reasoning loop
- Tool selection with many tools
- Structured outputs and validation
- Model Context Protocol (MCP)

See [2_tool_calls/tool_use.md](2_tool_calls/tool_use.md) for the detailed plan.

## Seminal ideas by topic

Each section aims to cover the foundational concepts that define its area. Here's the map of what we're building toward:

**Retrieval (Section 1):** Naive RAG, query rewriting, re-ranking, HyDE (Gao et al. 2022), agentic RAG, Recall@K evaluation. Future extensions: GraphRAG (Microsoft 2024), ColBERT late-interaction, RAPTOR hierarchical retrieval, Corrective RAG, Self-RAG, Adaptive RAG.

**Tool Calling (Section 2):** Function calling APIs, ReAct (Yao et al. 2022), Toolformer (Schick et al. 2023), Gorilla (Patil et al. 2023), parallel/sequential tool use, MCP (Anthropic 2024), structured outputs, tool selection at scale.

**Planning & Reasoning (Section 3):** Chain-of-thought (Wei et al. 2022), tree-of-thought (Yao et al. 2023), self-consistency (Wang et al. 2022), plan-and-execute, reflection/self-critique, graph-of-thought.

**Multi-Agent (Section 6):** Multi-agent debate (Du et al. 2023), agent specialization, hierarchical orchestration, swarm architectures, the handoff pattern.

**Memory (Section 5):** Working/short-term/long-term memory, episodic memory, Reflexion (Shinn et al. 2023), MemGPT virtual memory paging (Packer et al. 2023).

**Autonomous Agents (Section 7):** AutoGPT, BabyAGI, Voyager (Wang et al. 2023), goal decomposition, the agent = LLM + memory + planning + tools framework (Weng 2023).

**Protocols (Section 9):** Model Context Protocol (MCP), Agent-to-Agent Protocol (A2A, Google 2025), tool registries, JSON-RPC.

**Evaluation (Section 8):** LLM-as-judge, SWE-bench, WebArena, GAIA, AgentBench, Constitutional AI, NeMo Guardrails.

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

Run any script directly:

```bash
python 0_hello.py
python 1_rag/1.1_in_context_qa.py
python 1_rag/1_rag.py  # interactive TUI
```

## API access

| Provider | Env Variable | Used For |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | Direct API (embeddings, hello) |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-model access (eval, scoring, RAG) |

Both use the OpenAI SDK. OpenRouter is accessed by setting `base_url="https://openrouter.ai/api/v1"`.

## Documentation

The project includes a PDF report covering the theory, diagrams, and results for each section.

```bash
./docs/build.sh          # Build PDF -> docs/output/agents_report.pdf
./docs/build.sh --clean  # Remove generated PDF
```

Requires: pandoc, xelatex, Charter/Helvetica/DejaVu Sans Mono fonts.

## Philosophy

- **No frameworks.** Every pattern is implemented from scratch so you understand what's happening. `pip install langchain` teaches you nothing about how agents work.
- **One idea per file.** Each script isolates one technique. You can read them independently.
- **Measure everything.** Every technique gets benchmarked against baselines with real datasets. Opinions are cheap; numbers aren't.
- **Educational comments.** The code is heavily commented. It's meant to be read, not just run.
