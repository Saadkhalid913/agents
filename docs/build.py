#!/usr/bin/env python3
"""Build the agents report PDF from markdown chapters.

Usage:
    python docs/build.py          # Build PDF
    python docs/build.py --clean  # Remove generated PDF
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "agents_report.pdf")


def clean():
    """Remove the generated PDF."""
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Cleaned: {OUTPUT_FILE}")
    else:
        print(f"Nothing to clean: {OUTPUT_FILE} does not exist")


def build():
    """Build the PDF from markdown chapters."""
    # Collect markdown files in order
    md_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "[0-9]*.md")))

    if not md_files:
        print(f"Error: No markdown chapter files found in {SCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)

    print("Building PDF from:")
    for f in md_files:
        print(f"  {os.path.basename(f)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tex_file = os.path.join(OUTPUT_DIR, "agents_report.tex")

    # Step 1: Generate .tex with pandoc
    pandoc_cmd = [
        "pandoc",
        f"--metadata-file={os.path.join(SCRIPT_DIR, 'metadata.yaml')}",
        "--top-level-division=chapter",
        "--toc",
        "-N",
        f"--lua-filter={os.path.join(SCRIPT_DIR, 'filters', 'formatting.lua')}",
        "-s",
        "-o", tex_file,
        *md_files,
    ]

    print(f"Step 1: pandoc → .tex")
    result = subprocess.run(pandoc_cmd)
    if result.returncode != 0:
        print("\nPandoc failed.", file=sys.stderr)
        sys.exit(result.returncode)

    # Step 2: Compile .tex → .pdf with xelatex (two passes for TOC)
    # Run from OUTPUT_DIR so minted cache stays there
    xelatex_cmd = [
        "xelatex",
        "-shell-escape",
        "-interaction=nonstopmode",
        "agents_report.tex",
    ]

    for pass_num in (1, 2):
        print(f"Step 2.{pass_num}: xelatex pass {pass_num}")
        result = subprocess.run(xelatex_cmd, capture_output=True, text=True, cwd=OUTPUT_DIR)
        # xelatex returns non-zero for warnings too; check if PDF was produced
        if not os.path.exists(OUTPUT_FILE):
            log_file = os.path.join(OUTPUT_DIR, "agents_report.log")
            if os.path.exists(log_file):
                with open(log_file) as f:
                    lines = f.readlines()
                print("\n--- Last 30 lines of xelatex log ---")
                for line in lines[-30:]:
                    print(line, end="")
            else:
                print(result.stderr)
            print(f"\nxelatex pass {pass_num} failed.", file=sys.stderr)
            sys.exit(1)

    # Clean up auxiliary files and minted cache
    for ext in (".tex", ".aux", ".log", ".toc", ".out"):
        path = os.path.join(OUTPUT_DIR, f"agents_report{ext}")
        if os.path.exists(path):
            os.remove(path)
    minted_dir = os.path.join(OUTPUT_DIR, "_minted-agents_report")
    if os.path.exists(minted_dir):
        shutil.rmtree(minted_dir)

    print(f"\nPDF generated: {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Build the agents report PDF")
    parser.add_argument("--clean", action="store_true", help="Remove generated PDF")
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        build()


if __name__ == "__main__":
    main()
