"""
sitecustomize.py
Automatically imported by Python at startup if present on sys.path.

• Disables user-level site-packages (to keep NumPy-2 from leaking in)
• Removes any user-site paths that might have been pre-added
"""

import os, site, sys

# 1) Block user-site for THIS and all subprocesses
os.environ["PYTHONNOUSERSITE"] = "1"

# 2) Strip stray user-site paths that are already on sys.path
for p in list(sys.path):
    if "LocalCache" in p and "site-packages" in p:
        while p in sys.path:
            sys.path.remove(p)
