import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from langchain_core.tools import BaseTool

class InternalKnowledgeBaseTool(BaseTool):
    name: str = "internal_knowledge_base_search"
    description: str = (
        "Searches the internal knowledge base (an XML file representing codebase files) for relevant information. "
        "Input should be a concise query string relevant to the project structure, features, or deployment."
    )
    # The path should be relative to where internal_kb_tool.py is located.
    # From backend/src/agent/internal_kb_tool.py, ../../public/knowledge_base.xml leads to backend/public/knowledge_base.xml
    _kb_file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../public/knowledge_base.xml"
    )
    _knowledge_base_content: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        if not os.path.exists(self._kb_file_path):
            print(f"WARNING: Internal knowledge base file not found at {self._kb_file_path}")
            return

        try:
            tree = ET.parse(self._kb_file_path)
            root = tree.getroot()
            for file_elem in root.findall('file'):
                path = file_elem.get('path')
                content = file_elem.text.strip() if file_elem.text else ""
                if path:
                    self._knowledge_base_content[path] = content
            print(f"Loaded {len(self._knowledge_base_content)} entries from internal knowledge base.")
        except ET.ParseError as e:
            print(f"Error parsing internal knowledge base XML at {self._kb_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred loading internal KB from {self._kb_file_path}: {e}")

    def _run(self, query: str) -> str:
        """
        Use the tool to search the internal knowledge base.
        Performs a simple keyword search across file contents and paths.
        """
        if not self._knowledge_base_content:
            return "Internal knowledge base is not loaded or is empty."

        results = []
        query_lower = query.lower()

        # Simple search: find files where query is in content or path
        for path, content in self._knowledge_base_content.items():
            if query_lower in content.lower() or query_lower in path.lower():
                results.append(f"--- File: {path} ---\n{content}")

        if not results:
            return "No relevant information found in the internal knowledge base for your query."

        # Limit results to avoid overwhelming the LLM with too much context for simple example
        # In a real RAG system, you'd apply more sophisticated retrieval and chunking.
        return "\n\n".join(results[:3]) # Return up to 3 relevant files