#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
OUTPUT_FILE="${OUTPUT_DIR}/agents_report.pdf"

# Handle --clean flag
if [[ "${1:-}" == "--clean" ]]; then
    rm -f "${OUTPUT_FILE}"
    echo "Cleaned: ${OUTPUT_FILE}"
    exit 0
fi

# Collect markdown files in order
MD_FILES=$(find "${SCRIPT_DIR}" -maxdepth 1 -name '[0-9]*.md' | sort)

if [[ -z "${MD_FILES}" ]]; then
    echo "Error: No markdown chapter files found in ${SCRIPT_DIR}" >&2
    exit 1
fi

echo "Building PDF from:"
echo "${MD_FILES}" | while read -r f; do echo "  $(basename "$f")"; done

mkdir -p "${OUTPUT_DIR}"

pandoc \
    --metadata-file="${SCRIPT_DIR}/metadata.yaml" \
    --pdf-engine=xelatex \
    --top-level-division=chapter \
    --toc \
    -o "${OUTPUT_FILE}" \
    ${MD_FILES}

echo ""
echo "PDF generated: ${OUTPUT_FILE}"
