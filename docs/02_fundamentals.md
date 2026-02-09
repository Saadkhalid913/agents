# Fundamentals

## Hello World (`0_hello.py`)

Before you build anything complicated, you should verify the plumbing works. This script sends one message to an LLM and prints the response. If this doesn't work, nothing else will either.

```
.env (API key) --> 0_hello.py --> OpenAI API --> "Hello, world!"
```

That's it. Twenty-eight lines. The interesting stuff starts next.

## Measuring Quality (`1_rag/1.1_in_context_qa.py`)

Here's the first real question: how do you know if your model is any good?

You can't just eyeball a few answers. You need a benchmark --- a dataset with known questions and answers, and a systematic way to score responses. This script builds that evaluation framework from scratch, and every other script in the project reuses it.

The approach is called LLM-as-judge. A stronger model grades a weaker model's answers on four criteria: correctness, completeness, faithfulness, and clarity. Each criterion is worth 25 points, for a total of 0--100.

```
                    HotpotQA dataset
                         |
            question + all documents
                         |
                         v
    +-----------------------------------------+
    |  Eval model (kimi-k2.5)                 |
    |  "Answer this question using these docs"|
    +-----------------------------------------+
                         |
                   generated answer
                         |
                         v
    +-----------------------------------------+
    |  Scoring model (gemini-3-flash)         |
    |  "Grade this answer 0-100"              |
    +-----------------------------------------+
                         |
                   score + reasoning
```

The dataset is HotpotQA --- multi-hop questions that require combining facts from multiple documents. We evaluate 100 questions in parallel using 8 threads.

The key detail: we give the model *all* the documents. No retrieval, no searching. This is the control group. It tells us the ceiling --- how well the model does when it has everything it needs.

Run it:

```bash
python 1_rag/1.1_in_context_qa.py
python 1_rag/1.1_in_context_qa.py --eval-model openai/gpt-4o --num-examples 50
```
