"""
Streamlit Root Application Entry Point Wrapper.
Enables seamless execution for Streamlit Cloud and local runners (streamlit run app.py).
"""

import os
import sys

# Ensure src/ and app/ directories are in Python path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("app"))

# Execute app/app.py content
app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, "exec")
    exec(code, globals())
