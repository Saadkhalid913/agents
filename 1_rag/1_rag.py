import os
import sys
import subprocess
import json
import datetime
import questionary
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# RAG BENCHMARK ORCHESTRATOR (Interactive TUI)
# ============================================================================
# Interactive terminal UI for running RAG benchmarks. Pick a test suite,
# choose models from models.txt, set parameters, and get results saved
# as JSON.
#
# Two benchmark suites:
#   HotpotQA  — runs 1.1 (in-context), 1.2 (naive RAG), 1.3 (query reword)
#   BEIR/NQ   — runs 1.4 (large corpus RAG with 10K+ documents)
#
# Usage:
#   python 1_rag.py
# ============================================================================

MODELS_FILE = "models.txt"
RESULTS_FILE = "benchmark_results.json"

# HotpotQA subscripts: (filename, label, supports_rewrite)
HOTPOTQA_BENCHMARKS = [
    ("1.1_in_context_qa.py", "In-Context QA", False),
    ("1.2_naive_rag_with_embeddings.py", "Naive RAG", False),
    ("1.3_rag_with_query_rewording.py", "RAG + Query Reword", True),
]

# BEIR/NQ subscript
BEIR_BENCHMARK = ("1.4_large_corpus_rag.py", "Large Corpus RAG")


def load_models() -> list[str]:
    """Load model list from models.txt (one model ID per line)."""
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)
    if not os.path.exists(models_path):
        print(f"Error: {MODELS_FILE} not found. Create it with one model ID per line.")
        sys.exit(1)
    with open(models_path) as f:
        models = [line.strip() for line in f if line.strip()]
    if not models:
        print(f"Error: {MODELS_FILE} is empty.")
        sys.exit(1)
    return models


def run_benchmark(
    script: str,
    eval_model: str,
    scoring_model: str,
    num_examples: int,
    rewrite_model: str | None = None,
    corpus_size: int | None = None,
    top_k: int | None = None,
) -> dict | None:
    """
    Run a single benchmark script as a subprocess and capture its result.

    The subscript prints a RESULT_JSON:<json> line at the very end of its
    output, which we parse to extract the score summary.

    Args:
        script: Filename of the benchmark script
        eval_model: Model to evaluate
        scoring_model: Model for scoring answers
        num_examples: Number of dataset examples to evaluate
        rewrite_model: Model for query rewriting (only for 1.3)
        corpus_size: Number of documents in corpus (only for 1.4)
        top_k: Number of documents to retrieve (only for 1.4)

    Returns:
        Parsed result dict, or None if the script failed
    """
    cmd = [
        sys.executable, script,
        "--eval-model", eval_model,
        "--scoring-model", scoring_model,
        "--num-examples", str(num_examples),
    ]
    if rewrite_model:
        cmd.extend(["--rewrite-model", rewrite_model])
    if corpus_size is not None:
        cmd.extend(["--corpus-size", str(corpus_size)])
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])

    print(f"  Running: {' '.join(cmd)}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )

    output_lines = []
    for line in process.stdout:
        print(f"    {line}", end="")
        output_lines.append(line.rstrip())

    process.wait()

    if process.returncode != 0:
        print(f"\n  [ERROR] {script} exited with code {process.returncode}\n")
        return None

    # Find and parse the RESULT_JSON line
    for line in reversed(output_lines):
        if line.startswith("RESULT_JSON:"):
            try:
                return json.loads(line[len("RESULT_JSON:"):])
            except json.JSONDecodeError:
                print(f"\n  [ERROR] Failed to parse result JSON from {script}\n")
                return None

    print(f"\n  [ERROR] No RESULT_JSON line found in {script} output\n")
    return None


def print_comparison(results: list[tuple[str, dict | None]]) -> None:
    """Print a side-by-side comparison table of benchmark results."""
    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    # Check if any results have recall (BEIR)
    has_recall = any(
        r is not None and "avg_recall_at_k" in r for _, r in results)

    # Header
    header = f"{'Approach':<30} {'Score':>8} {'Examples':>10}"
    if has_recall:
        header += f" {'Recall@K':>10}"
    print(header)
    print("-" * 70)

    for label, result in results:
        if result is None:
            print(f"{label:<30} {'FAILED':>8} {'--':>10}")
        else:
            score = result["overall_score"]
            n = result["num_examples"]
            line = f"{label:<30} {score:>7.2f} {n:>10}"
            if has_recall and "avg_recall_at_k" in result:
                line += f" {result['avg_recall_at_k']:>9.4f}"
            print(line)

    print("-" * 70)

    # Score distribution breakdown
    successful = [(label, r) for label, r in results if r is not None]
    if successful:
        print(f"\n{'Score Distribution:':<30}", end="")
        for label, _ in successful:
            print(f" {label:>20}", end="")
        print()
        print("-" * 70)

        dist_keys = ["90-100", "80-89", "70-79", "60-69", "below-60"]
        for key in dist_keys:
            print(f"  {key:<28}", end="")
            for _, r in successful:
                count = r["score_distribution"].get(key, 0)
                print(f" {count:>20}", end="")
            print()

    print("=" * 70)


def save_results(
    suite: str,
    config: dict,
    results: list[tuple[str, dict | None]],
) -> None:
    """Append benchmark results to the JSON results file."""
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), RESULTS_FILE)

    # Load existing results
    existing = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "suite": suite,
        "config": config,
        "benchmarks": {
            label: result for label, result in results if result is not None
        },
    }

    existing.append(entry)

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")


def run_hotpotqa_suite(models: list[str]) -> None:
    """Interactive flow for HotpotQA benchmarks (1.1, 1.2, 1.3)."""
    print("\n--- HotpotQA Benchmark Suite ---\n")

    eval_model = questionary.select(
        "Eval model:", choices=models).ask()
    if not eval_model:
        return

    scoring_model = questionary.select(
        "Scoring model:", choices=models).ask()
    if not scoring_model:
        return

    rewrite_model = questionary.select(
        "Rewrite model (for 1.3):", choices=models).ask()
    if not rewrite_model:
        return

    num_examples = questionary.text(
        "Number of examples:", default="100").ask()
    if not num_examples:
        return
    num_examples = int(num_examples)

    config = {
        "eval_model": eval_model,
        "scoring_model": scoring_model,
        "rewrite_model": rewrite_model,
        "num_examples": num_examples,
    }

    print(f"\n{'=' * 70}")
    print("HOTPOTQA BENCHMARK SUITE")
    print(f"{'=' * 70}")
    print(f"Eval model:    {eval_model}")
    print(f"Scoring model: {scoring_model}")
    print(f"Rewrite model: {rewrite_model}")
    print(f"Num examples:  {num_examples}")
    print(f"{'=' * 70}\n")

    # Run benchmarks in parallel
    results_map = {}
    with ThreadPoolExecutor(max_workers=len(HOTPOTQA_BENCHMARKS)) as executor:
        future_to_label = {
            executor.submit(
                run_benchmark,
                script,
                eval_model,
                scoring_model,
                num_examples,
                rewrite_model if supports_rewrite else None,
            ): label
            for script, label, supports_rewrite in HOTPOTQA_BENCHMARKS
        }

        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                results_map[label] = future.result()
            except Exception as e:
                print(f"\n  [ERROR] '{label}' failed: {e}\n")
                results_map[label] = None

    # Maintain original order
    all_results = [(label, results_map.get(label))
                   for _, label, _ in HOTPOTQA_BENCHMARKS]

    print()
    print_comparison(all_results)
    save_results("hotpotqa", config, all_results)


def run_beir_suite(models: list[str]) -> None:
    """Interactive flow for BEIR/NQ benchmark (1.4)."""
    print("\n--- BEIR/NQ Benchmark Suite ---\n")

    eval_model = questionary.select(
        "Eval model:", choices=models).ask()
    if not eval_model:
        return

    scoring_model = questionary.select(
        "Scoring model:", choices=models).ask()
    if not scoring_model:
        return

    num_examples = questionary.text(
        "Number of examples:", default="50").ask()
    if not num_examples:
        return
    num_examples = int(num_examples)

    corpus_size = questionary.text(
        "Corpus size (num documents):", default="10000").ask()
    if not corpus_size:
        return
    corpus_size = int(corpus_size)

    top_k = questionary.text(
        "Top-K retrieval:", default="5").ask()
    if not top_k:
        return
    top_k = int(top_k)

    config = {
        "eval_model": eval_model,
        "scoring_model": scoring_model,
        "num_examples": num_examples,
        "corpus_size": corpus_size,
        "top_k": top_k,
    }

    script, label = BEIR_BENCHMARK

    print(f"\n{'=' * 70}")
    print("BEIR/NQ BENCHMARK SUITE")
    print(f"{'=' * 70}")
    print(f"Eval model:    {eval_model}")
    print(f"Scoring model: {scoring_model}")
    print(f"Corpus size:   {corpus_size:,}")
    print(f"Top-K:         {top_k}")
    print(f"Num examples:  {num_examples}")
    print(f"{'=' * 70}\n")

    result = run_benchmark(
        script, eval_model, scoring_model, num_examples,
        corpus_size=corpus_size, top_k=top_k,
    )

    all_results = [(label, result)]
    print()
    print_comparison(all_results)
    save_results("beir_nq", config, all_results)


if __name__ == "__main__":
    models = load_models()

    print("=" * 70)
    print("RAG BENCHMARK ORCHESTRATOR")
    print("=" * 70)
    print()

    suite = questionary.select(
        "Which benchmark suite?",
        choices=[
            questionary.Choice("HotpotQA Tests (1.1, 1.2, 1.3)", value="hotpotqa"),
            questionary.Choice("BEIR/NQ Tests (1.4)", value="beir"),
        ],
    ).ask()

    if suite == "hotpotqa":
        run_hotpotqa_suite(models)
    elif suite == "beir":
        run_beir_suite(models)
