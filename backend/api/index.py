# backend/api/index.py
import sys
import os

# Add the 'src' directory to the Python path to allow importing 'agent.app'
# Vercel places the 'api' directory at the root of the serverless function's deployment.
# So, to find 'src', we go up one level ('..') from this file's directory (which is 'api')
# to the backend root, and then into 'src'.
current_dir = os.path.dirname(__file__)
backend_root = os.path.abspath(os.path.join(current_dir, '..'))
src_path = os.path.join(backend_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agent.app import app
except ImportError as e:
    additional_info = f"Current sys.path: {sys.path}. CWD: {os.getcwd()}. Files in backend_root ({backend_root}): {os.listdir(backend_root) if os.path.exists(backend_root) else 'N/A'}. Files in src_path ({src_path}): {os.listdir(src_path) if os.path.exists(src_path) else 'N/A'}."
    raise ImportError(
        f"Failed to import 'app' from 'agent.app'. {additional_info} Error: {e}"
    )

