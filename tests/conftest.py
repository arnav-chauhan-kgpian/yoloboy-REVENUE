"""
tests/conftest.py
==================
Ensure the project root is on PYTHONPATH so that ``src.*`` and
``streamlit_app.*`` imports resolve correctly when pytest is invoked from
any working directory.
"""

import sys
from pathlib import Path

# Insert project root (parent of tests/) at the front of sys.path.
# This is a no-op if the root is already present (e.g. when running from
# the project root with ``pytest tests/``).
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
