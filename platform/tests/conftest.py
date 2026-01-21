# Ensure repo root is on sys.path for absolute imports like `platform.services.*`
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
