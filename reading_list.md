# Reading List

Papers and resources organized by the topics explored in this project. Start with the foundational papers, then follow the sections that match your interests.

---

## Retrieval-Augmented Generation

These cover the core RAG pipeline (sections 1.1-1.8).

**Foundational:**

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
  https://arxiv.org/abs/2005.11401
  _The original RAG paper. Combines a pretrained seq2seq model with a retrieval component trained end-to-end._

- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (2020)
  https://arxiv.org/abs/2004.04906
  _DPR -- learned dense embeddings for retrieval instead of BM25. The backbone behind most modern RAG systems._

**Query Rewriting (section 1.3, 1.5):**

- Ma et al., "Query Rewriting for Retrieval-Augmented Large Language Models" (2023)
  https://arxiv.org/abs/2305.14283
  _Formalizes the idea that user queries are suboptimal for retrieval and proposes trainable rewriting._

**Re-ranking (section 1.6):**

- Nogueira & Cho, "Passage Re-ranking with BERT" (2019)
  https://arxiv.org/abs/1901.04085
  _Early work on neural re-ranking -- shows that a cross-encoder dramatically improves retrieval precision._

- Sun et al., "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (2023)
  https://arxiv.org/abs/2304.09542
  _LLMs as re-rankers (what we do in 1.6). Compares listwise, pointwise, and pairwise prompting strategies._

**HyDE (section 1.7):**

- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
  https://arxiv.org/abs/2212.10496
  _The HyDE paper. Generate a hypothetical document, embed it, and use that for retrieval. Surprisingly effective._

**Agentic RAG (section 1.8):**

- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2023)
  https://arxiv.org/abs/2310.11511
  _Model decides when to retrieve and critiques its own output -- the trained version of our agentic loop._

- Jiang et al., "Active Retrieval Augmented Generation" (2023)
  https://arxiv.org/abs/2305.06983
  _FLARE -- forward-looking active retrieval. Model retrieves on-demand when it detects low confidence._

**Surveys:**

- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2024)
  https://arxiv.org/abs/2312.10997
  _Comprehensive survey covering naive RAG, advanced RAG, and modular RAG. Good map of the whole landscape._

---

## Embeddings & Vector Search

Background on the retrieval infrastructure used throughout section 1.

- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
  https://arxiv.org/abs/1908.10084
  _How to turn BERT into a sentence encoder. Foundation for all modern embedding models._

- Neelakantan et al., "Text and Code Embeddings by Contrastive Pre-Training" (2022)
  https://arxiv.org/abs/2201.10005
  _OpenAI's embedding approach (ancestor of text-embedding-3-small that we use). Contrastive learning at scale._

- Johnson et al., "Billion-scale similarity search with GPUs" (2017)
  https://arxiv.org/abs/1702.08734
  _The FAISS paper. Understanding approximate nearest neighbor search matters when you scale past toy corpora._

---

## Evaluation & Benchmarks

The evaluation framework used across all scripts.

- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
  https://arxiv.org/abs/2306.05685
  _Formalizes LLM-as-judge (our scoring approach). Analyzes biases: position, verbosity, self-enhancement._

- Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering" (2018)
  https://arxiv.org/abs/1809.09600
  _The dataset behind scripts 1.1-1.3. Multi-hop questions that require reasoning across documents._

- Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (2021)
  https://arxiv.org/abs/2104.08663
  _The benchmark framework behind notebooks 1.4-1.8. Standardized evaluation across diverse retrieval tasks._

---

## Tool Use & Function Calling

Covers the mechanics explored in section 2.

- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
  https://arxiv.org/abs/2302.04761
  _Self-supervised tool learning. Model learns when and how to call APIs by training on its own annotations._

- Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" (2023)
  https://arxiv.org/abs/2307.16789
  _Scaling tool use to thousands of APIs. Introduces ToolBench and decision-tree reasoning for API selection._

- Patil et al., "Gorilla: Large Language Model Connected with Massive APIs" (2023)
  https://arxiv.org/abs/2305.15334
  _Fine-tuning for accurate API calls. Addresses hallucination in function signatures and parameters._

---

## Agents & Reasoning

These bridge what you've built toward the planned sections (3-7).

**Reasoning (section 3 preview):**

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
  https://arxiv.org/abs/2201.11903
  _The chain-of-thought paper. "Let's think step by step" unlocks multi-step reasoning._

- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
  https://arxiv.org/abs/2305.10601
  _Generalize CoT to a tree -- explore multiple reasoning paths, backtrack, and select the best._

- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022)
  https://arxiv.org/abs/2203.11171
  _Sample multiple CoT paths and take the majority vote. Simple ensemble over reasoning chains._

- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
  https://arxiv.org/abs/2303.11366
  _Agents that reflect on failures and improve. Verbal self-feedback as a learning signal._

**Agents (sections 6-7 preview):**

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
  https://arxiv.org/abs/2210.03629
  _Interleave thinking and acting. The pattern behind most modern agent loops (reason, act, observe, repeat)._

- Significant Gravitas, "AutoGPT" (2023)
  https://github.com/Significant-Gravitas/AutoGPT
  _Not a paper but worth studying. Goal-driven autonomous agent with task decomposition and self-prompting._

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023)
  https://arxiv.org/abs/2304.03442
  _Multi-agent simulation with memory, reflection, and planning. 25 agents living in a virtual town._

- Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)
  https://arxiv.org/abs/2308.08155
  _Framework for multi-agent conversations. Relevant to planned section 6 on multi-agent systems._

---

## Memory (section 5 preview)

- Zhang et al., "A Survey on the Memory Mechanism of Large Language Model based Agents" (2024)
  https://arxiv.org/abs/2404.13501
  _Covers short-term, long-term, episodic, and semantic memory architectures for LLM agents._

---

## Suggested Reading Order

If reading sequentially, this order follows the project's arc:

1. Lewis et al. (RAG) -- the big picture
2. Karpukhin et al. (DPR) -- how retrieval actually works
3. Gao et al. (HyDE) -- creative retrieval thinking
4. Asai et al. (Self-RAG) -- retrieval meets agency
5. Zheng et al. (LLM-as-Judge) -- how we evaluate everything
6. Schick et al. (Toolformer) -- foundation for tool use
7. Yao et al. (ReAct) -- the agent loop pattern
8. Wei et al. (Chain-of-Thought) -- reasoning foundations
9. Shinn et al. (Reflexion) -- agents that learn from mistakes
10. Park et al. (Generative Agents) -- where it all goes
