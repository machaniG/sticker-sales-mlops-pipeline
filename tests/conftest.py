import sys
from pathlib import Path

# Get the path to the project root (one level up from the 'tests' directory)
# This directory contains the 'scripts' package.
project_root = Path(__file__).parent.parent

# Insert the project root path at the beginning of sys.path
# This ensures that packages like 'scripts' can be imported by the tests.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Added project root to sys.path: {project_root}")