# Section 2: Tool Calling — Brainstorm

Planning document for the tool calling section. Each subsection maps to a script or notebook.

## The arc

Start with the simplest possible tool call (one function, one round-trip), then layer on complexity: parallel calls, chained calls, error handling, dynamic tool discovery, and finally the ReAct loop that ties it all together.

The progression mirrors Section 1: controlled experiment first (single tool), then scale up the problem (many tools, multi-step tasks), then add intelligence (the model decides what to do).

---

## 2.1 — Single Tool Call (the hello world)

The absolute minimum: define one tool as a JSON schema, send it to the model, get back a structured tool_call, execute it, return the result.

**What to build:**
- Define a `get_weather(city: str)` tool with JSON Schema
- Send a user message + tool definition to the API
- Parse the model's tool_call response
- Execute the function locally
- Feed the result back as a `tool` message
- Get the final natural-language response

**What this teaches:**
- The 5-step lifecycle: context assembly, model decision, structured output, execution, result injection
- JSON Schema for tool definitions (name, description, parameters, required fields)
- The `tool_choice` parameter: `auto`, `required`, `none`, forced specific tool
- How tool results get injected back (role: "tool", tool_call_id linking)

**Key concepts:** function calling API, tool schemas, structured outputs

---

## 2.2 — Parallel Tool Calls

Multiple independent tools called in a single model turn.

**What to build:**
- Define 3-4 tools (weather, stock price, news headlines, calculator)
- Ask a question that requires multiple tools ("What's the weather in NYC and the current price of AAPL?")
- Model returns multiple tool_calls in one response
- Execute all in parallel (ThreadPoolExecutor)
- Return all results, get final answer

**What this teaches:**
- Parallel vs sequential execution (4x300ms API calls in 300ms instead of 1.2s)
- How the model decides which tools to call simultaneously
- Handling multiple tool results in one conversation turn

**Key concepts:** parallel function calling, concurrent execution

---

## 2.3 — Chained / Sequential Tool Calls

Output of one tool feeds into another. Multi-turn tool use.

**What to build:**
- Tools that depend on each other: `search_database(query)` returns IDs, `get_details(id)` returns full records
- The model calls tool A, gets the result, then decides to call tool B with data from tool A's result
- Track the multi-turn conversation: user -> assistant(tool_call) -> tool(result) -> assistant(tool_call) -> tool(result) -> assistant(final answer)

**What this teaches:**
- State management across tool call turns
- How the model reasons about intermediate results
- The conversation structure for multi-step tool use
- When to let the model drive the chain vs when to hardcode the pipeline

**Key concepts:** multi-turn tool use, state threading, conversation management

---

## 2.4 — Error Handling and Retries

What happens when tools fail, and how the model recovers.

**What to build:**
- Tools that sometimes fail (simulated network errors, rate limits, invalid inputs)
- Return structured error messages to the model (not just crash)
- Let the model decide: retry? try a different tool? answer without the tool?
- Implement exponential backoff for transient failures
- Compare model behavior with good vs bad error messages

**What this teaches:**
- Error handling patterns for tool-calling agents
- How error message quality affects model recovery
- The retry vs fallback vs give-up decision
- Structured error responses that help the model reason about failures

**Key concepts:** error handling, graceful degradation, retry strategies

---

## 2.5 — ReAct Loop (Reasoning + Acting)

The foundational agent pattern. The model alternates between thinking and acting until the task is done.

**What to build:**
- Implement the Thought -> Action -> Observation loop from scratch
- Give the model a set of tools and a complex question requiring multiple steps
- The model reasons about what to do (Thought), calls a tool (Action), sees the result (Observation), then decides next steps
- Run until the model decides it has enough information to answer
- Track and display the full reasoning trace

**What this teaches:**
- The ReAct pattern (Yao et al., 2022) — the most influential agent architecture
- How to implement an open-ended agent loop with a termination condition
- Reasoning traces for transparency and debugging
- The difference between a pipeline (fixed steps) and an agent (model-driven steps)

**Seminal paper:** ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023)

**Key concepts:** ReAct, agent loop, reasoning traces, autonomous tool selection

---

## 2.6 — Tool Selection at Scale (many tools)

What happens when you have 20+ tools and the model has to pick the right ones.

**What to build:**
- Define 20-30 tools spanning multiple domains (math, search, file ops, data analysis, communication)
- Measure tool selection accuracy: does the model pick the right tool?
- Implement embedding-based tool retrieval: embed tool descriptions, retrieve top-K relevant tools per query, present only those to the model
- Compare: all tools in context vs retrieved subset
- Benchmark: accuracy, latency, token cost

**What this teaches:**
- Tool selection degrades with too many tools in context
- Retrieval-based tool filtering (Anthropic's Tool Search Tool concept)
- The tradeoff between tool availability and selection accuracy
- Token economics: 30 tool schemas can consume 10K+ tokens

**Key concepts:** tool retrieval, dynamic tool loading, context window management

---

## 2.7 — Structured Outputs and Validation

Guaranteeing that tool calls conform to their schemas.

**What to build:**
- Use strict mode / structured outputs to guarantee schema compliance
- Define tools with complex nested schemas (arrays of objects, enums, optional fields)
- Implement Pydantic-based validation of tool call arguments
- Compare strict vs non-strict: how often does the model hallucinate fields or miss required params?
- Add input_examples to tool definitions and measure accuracy improvement

**What this teaches:**
- Structured outputs / strict mode (constrained decoding)
- Schema validation with Pydantic
- Tool use examples as few-shot guidance
- The gap between "valid JSON" and "correct usage"

**Key concepts:** structured outputs, schema validation, constrained decoding, tool use examples

---

## 2.8 — MCP: Model Context Protocol

Build an MCP server and client from scratch.

**What to build:**
- Implement a simple MCP server that exposes 3-4 tools via the JSON-RPC protocol
- Implement an MCP client that discovers tools from the server at runtime
- Wire the client into an agent loop: discover tools -> present to model -> execute via server -> return results
- Show dynamic tool registration: add a new tool to the server, client picks it up without restart

**What this teaches:**
- The MCP architecture: hosts, clients, servers
- JSON-RPC 2.0 message format
- Dynamic tool discovery vs static tool definitions
- The "USB-C of AI" — standardized tool integration
- How MCP differs from hardcoded function calling

**Key concepts:** Model Context Protocol, JSON-RPC, tool discovery, interoperability

---

## Evaluation approach

Unlike Section 1 (where we had ground-truth answers to score), tool calling evaluation is about:

1. **Tool selection accuracy** — did the model pick the right tool(s)?
2. **Argument correctness** — are the parameters valid and appropriate?
3. **Task completion** — did the multi-step plan achieve the goal?
4. **Efficiency** — how many tool calls / tokens / rounds did it take?

For scripts with measurable outcomes (2.1-2.4, 2.6-2.7), we can build automated benchmarks. For the ReAct loop (2.5) and MCP (2.8), evaluation is more qualitative — trace inspection and task completion rates.

---

## Seminal papers to reference

| Paper | Year | Key Idea |
|-------|------|----------|
| ReAct (Yao et al.) | 2022 | Thought-Action-Observation loop |
| Toolformer (Schick et al.) | 2023 | Self-supervised tool learning |
| Gorilla (Patil et al.) | 2023 | Fine-tuned LLaMA beats GPT-4 on API calling |
| ToolACE | 2024 | Synthetic training data for tool use |
| Tool Learning Survey | 2024 | Four-stage taxonomy: plan, select, call, respond |
| MCP Specification | 2024-2025 | Open protocol for tool integration |

---

## Open questions

- Should 2.5 (ReAct) use the HotpotQA dataset from Section 1 for continuity? That would let us compare ReAct+tools vs pure RAG on the same questions.
- How much of MCP (2.8) is protocol implementation vs agent behavior? Could split into 2.8 (build server) and 2.9 (build client + agent).
- Is there a good benchmark dataset for tool selection accuracy? The Berkeley Function Calling Leaderboard (BFCL) has evaluation data we could adapt.
- Should we cover plan-then-execute as a contrast to ReAct? Could be a 2.9 or save for the planning section.
