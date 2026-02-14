Scaffold a new agent file for: $ARGUMENTS

Instructions:
1. List existing agent files matching the pattern `[0-9]*.py` in the project root
2. Determine the next sequential number (e.g., if `2_*.py` exists, the next is `3`)
3. Convert the argument description to a snake_case filename: `N_description.py`
4. Create the new Python file with this structure:

```python
import os
import json
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# ============================================================================
# [TITLE IN CAPS]
# ============================================================================
# [Multi-line comment block explaining what this agent does,
#  what paradigm it demonstrates, and key concepts]
# ============================================================================

# Configuration
# [Model names, API setup, constants]


def main():
    """[Docstring with Google-style Args/Returns]"""
    pass


if __name__ == "__main__":
    main()
```

5. Follow these conventions:
   - Educational comments explaining what and why
   - Google-style docstrings with Args/Returns
   - Type hints on all function signatures
   - Constants (model names, config) at the top of the file
   - Self-contained: the file should run with `python N_name.py`
   - Use OpenRouter for LLM access unless direct OpenAI is specifically needed
6. After creating the file, remind the user to:
   - Implement the agent logic
   - Document it with `/document-agent N_name.py`
   - Add any new dependencies to `requirements.txt`
