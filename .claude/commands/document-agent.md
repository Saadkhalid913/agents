Document the agent file: $ARGUMENTS

Instructions:
1. Read the Python file specified in the argument (e.g., `2_naive_rag_with_embeddings.py`)
2. Determine which chapter it belongs to based on its paradigm and the existing chapter structure in `docs/`
3. Add a new `##` section to the appropriate chapter file following this template:

```
## Title (`N_filename.py`)

### What It Does
[High-level description — one paragraph]

### Key Concepts
[Bulleted list of concepts with definitions]

### Architecture
[ASCII diagram showing data flow]

### Implementation Details
[Walkthrough of the code with key snippets]

### Configuration
[Table: Parameter | Value | Notes]

### Running It
[Command to run + expected output description]
```

4. Read the existing chapter file first to match its style and voice
5. Include actual code snippets from the Python file (not pseudocode)
6. After writing, remind the user to run `./docs/build.sh` to rebuild the PDF
