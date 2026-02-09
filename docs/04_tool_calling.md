# Tool Calling

Tool calling lets an LLM invoke external functions during a conversation --- search the web, query a database, run code. Instead of generating text, the model outputs a structured request to call a specific function. You execute it and feed the result back. This loop is what turns a chatbot into an agent.

The key insight: the model never executes anything. It reads a JSON schema describing a function, decides when to call it, and emits structured arguments. Your code does the execution. This separation is what makes the pattern safe and composable.

## The Lifecycle (`2_tool_calls/2.1_single_tool_call.ipynb`)

Every tool call follows the same five-step lifecycle:

```
   User message + tool schemas
              |
   +----------v-----------+
   |  LLM reasons about    |
   |  whether to call a    |
   |  tool and which one   |
   +----------+------------+
              |
   +----------v-----------+
   |  Structured output:   |
   |  { "name": "fn",      |
   |    "arguments": {...}} |
   +----------+------------+
              |
   +----------v-----------+
   |  YOUR CODE executes   |
   |  the function locally |
   +----------+------------+
              |
   +----------v-----------+
   |  Result injected back |
   |  into conversation    |
   +----------+------------+
              |
   +----------v-----------+
   |  LLM generates final  |
   |  natural-language      |
   |  response              |
   +----------------------+
```

A tool is defined as a JSON schema with three parts: a name, a description (which the model reads to decide *when* to call it), and a parameters schema (which tells the model *how* to call it):

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["city"],
        },
    },
}]
```

The implementation lives in your code. The model never sees it:

```python
def get_weather(city: str) -> dict:
    return {"temp_f": 62, "conditions": "Foggy"}

available_functions = {"get_weather": get_weather}
```

The lifecycle then plays out in two API calls:

```python
# First call: model decides to use a tool
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools,
)
tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name = "get_weather"
# tool_call.function.arguments = '{"city": "San Francisco"}'

# Execute locally
result = get_weather(**json.loads(tool_call.function.arguments))

# Second call: feed the result back
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result),
})
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools,
)
# response.choices[0].message.content = "The weather in San
#   Francisco is foggy with a temperature of 62 F..."
```

The `tool_call_id` links the result to the specific request. This is important when there are multiple tool calls --- it tells the model which result corresponds to which call.

Here's the full conversation trace for a single tool call:

```
[0] role=user
    content: What's the weather in London?

[1] role=assistant
    tool_call: get_weather({"city":"London"})

[2] role=tool  tool_call_id=tool_get_weather_FP46U9...
    content: {"temp_f": 50, "conditions": "Rainy", "humidity": 85}

[3] role=assistant
    content: The weather in London is currently rainy with a
             temperature of 50 F and 85% humidity.
```

Four messages: user question, assistant tool request, tool result, assistant final answer. Two API round-trips. One local function execution in between.

When the model doesn't need a tool, it skips the whole mechanism and responds with text directly. The `tool_calls` field is `None`, and you get the answer in one round-trip.

## Parallel Tool Calls (`2_tool_calls/2.2_parallel_tool_calls.ipynb`)

When a question requires multiple independent pieces of information, the model can request several tool calls in a single turn:

```
   "What's the weather in Tokyo,
    AAPL's stock price,
    and the latest AI news?"
              |
   +----------v-----------+
   |  Model returns 3      |
   |  tool_calls at once   |
   +----------+------------+
              |
   +---+------+------+----+
   |   |             |    |
   v   v             v    |
  get  get_stock  get_news|
  weather  price  headlines
   |   |             |    |
   +---+------+------+----+
              |
   +----------v-----------+
   |  All 3 results sent   |
   |  back together        |
   +----------+------------+
              |
   +----------v-----------+
   |  Final answer using   |
   |  all three results    |
   +----------------------+
```

The model identifies which parts of the question are independent and batches them. You execute all calls concurrently:

```python
def execute_tool_calls_parallel(tool_calls):
    def execute_one(tc):
        fn = available_functions[tc.function.name]
        args = json.loads(tc.function.arguments)
        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(fn(**args)),
        }

    with ThreadPoolExecutor(max_workers=len(tool_calls)) as ex:
        return list(ex.map(execute_one, tool_calls))
```

The speedup is significant. With three tools that each take 300ms:

```
3 tools (each 300ms):
  Parallel:   0.305s
  Sequential: 0.916s
  Speedup:    3.0x
```

This is a latency optimization, not a capability one --- you get the same answer either way. But in production, where tools hit real APIs with real latency, the difference between 300ms and 900ms matters.

## Chained Tool Calls (`2_tool_calls/2.3_chained_tool_calls.ipynb`)

Not all tool calls are independent. Sometimes the output of one feeds into the next. The model has to search for a user (to get their ID), then use that ID to look up their profile or orders. It can't skip ahead.

This is multi-turn tool use. The conversation grows with each round-trip:

```
   "How much has Carol Davis spent?"
              |
   +----------v-----------+
   |  Round 1: search for  |
   |  "Carol Davis"         |
   +----------+------------+
              |
   search_users("Carol Davis")
   --> {"user_id": "u_103"}
              |
   +----------v-----------+
   |  Round 2: get orders  |
   |  for user u_103        |
   +----------+------------+
              |
   get_user_orders("u_103")
   --> {"total_spent": 174.97}
              |
   +----------v-----------+
   |  Round 3: final answer|
   |  "Carol has spent      |
   |   $174.97 on orders"  |
   +----------------------+
```

The implementation is a loop that keeps calling the API until the model stops requesting tools:

```python
def run_chained(user_message, max_rounds=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for round_num in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools,
        )
        assistant_msg = response.choices[0].message

        # No tool calls = final answer
        if not assistant_msg.tool_calls:
            return assistant_msg.content

        # Execute all tool calls for this round
        messages.append(assistant_msg)
        for tc in assistant_msg.tool_calls:
            fn = available_functions[tc.function.name]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })
```

The interesting thing is that chaining and parallelism compose. When the model needs both a profile and order history after a search, it calls both in the same round:

```
Round 1: search_users("Bob Smith")
  --> {"user_id": "u_102"}

Round 2: get_user_profile("u_102")     [parallel]
         get_user_orders("u_102")      [parallel]
  --> profile + orders

Round 3: "Bob Smith has the Basic plan. His total
          spending is $129.99..."
```

Search is sequential (can't call profile without the user ID), but profile and orders are parallel (both just need the ID). The model figures this out on its own.

### Conversation growth

Each round adds messages. A three-round chained query produces 8 messages:

```
[0] system:    "You are a helpful assistant..."
[1] user:      "What plan is Alice on and what are her orders?"
[2] assistant: tool_calls: [search_users]
[3] tool:      {"results": [{"user_id": "u_101", ...}]}
[4] assistant: tool_calls: [get_user_profile, get_user_orders]
[5] tool:      {"user_id": "u_101", "plan": "Premium", ...}
[6] tool:      {"user_id": "u_101", "orders": [...], ...}
[7] assistant: "Alice is on the Premium plan. Her orders..."
```

This is the fundamental tradeoff of chained tool calls: each round adds context that helps the model make better decisions, but also consumes more tokens. For complex multi-step tasks, the conversation can grow quickly.

## The system prompt matters

One subtle finding: without a system prompt telling the model to use tools proactively, smaller models tend to ask clarifying questions instead of chaining. They'll search for a user, see the result, and respond "I found Alice. Would you like to see her profile?" instead of just calling `get_user_profile`.

The fix is simple --- tell the model what you expect:

```python
SYSTEM_PROMPT = """You are a helpful assistant with access to
a user database. When a user asks about a person, ALWAYS use
the available tools to find the answer. Do not ask clarifying
questions -- use search_users to find the user, then use
get_user_profile or get_user_orders to get the details they
asked about."""
```

This is a general lesson: tool definitions tell the model *what it can do*, but the system prompt tells it *how aggressively to do it*. Both matter.

## What we learned

**2.1** established the core lifecycle. Two API calls, one local function execution, four messages in the conversation. The model reads schemas, not code.

**2.2** showed that independent tool calls can be batched and executed concurrently. The 3x speedup (0.3s vs 0.9s) is free --- same answer, less wall time.

**2.3** demonstrated that the model can drive multi-step workflows by chaining tool calls across rounds. It even combines chaining with parallelism when it recognizes independent sub-tasks within a round.

The progression from here: error handling (what happens when tools fail?), the ReAct reasoning loop (thinking before acting), tool selection at scale (picking from 20+ tools), and the Model Context Protocol (standardizing tool integration across systems).
