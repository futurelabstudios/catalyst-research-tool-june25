import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from agent.configuration import Configuration # Assuming Configuration is in agent.configuration

class InternalKnowledgeBaseTool(BaseTool):
    name: str = "internal_knowledge_base_search"
    description: str = (
        "Selects a relevant internal knowledge base file and provides its full content "
        "to the LLM based on the user's query."
        "Input should be the name of the chosen knowledge base topic (e.g., 'gender', 'climate')."
        "Note: The full content of the selected file is passed to the LLM, possibly truncated."
    )
    # CHANGE: This is now a directory path, not a file path
    _kb_directory_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../public/" 
    )
    # Stores map of topic name (e.g., "gender") to full file path (e.g., ".../extracted_gender.xml")
    _available_kb_files: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        print(f"InternalKnowledgeBaseTool initialized. KB directory path: {self._kb_directory_path}")
        self._scan_knowledge_base_files()
        if self._available_kb_files:
            print(f"Found {len(self._available_kb_files)} internal KB topics.")
            print(f"Available topics: {', '.join(self._available_kb_files.keys())}")
        else:
            print(f"WARNING: No internal KB XML files (extracted_*.xml) found in {self._kb_directory_path}.")

    def _scan_knowledge_base_files(self):
        """Scans the _kb_directory_path for 'extracted_*.xml' files and populates _available_kb_files."""
        self._available_kb_files = {}
        if not os.path.isdir(self._kb_directory_path):
            print(f"WARNING: Internal knowledge base directory not found at {self._kb_directory_path}")
            return

        try:
            for filename in os.listdir(self._kb_directory_path):
                if filename.startswith("extracted_") and filename.endswith(".xml"):
                    # Extract topic name from filename (e.g., "extracted_gender.xml" -> "gender")
                    topic_name = filename[len("extracted_"):-len(".xml")]
                    full_path = os.path.join(self._kb_directory_path, filename)
                    self._available_kb_files[topic_name] = full_path
                    print(f"Discovered KB topic: '{topic_name}' at '{full_path}'")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred scanning KB directory {self._kb_directory_path}: {e}")

    def get_available_topics_for_llm(self) -> str:
        """Returns a formatted string of available topics for the LLM to choose from."""
        if not self._available_kb_files:
            return "No internal knowledge base topics are available."
        
        topics = sorted(self._available_kb_files.keys())
        return "Available knowledge base topics: " + ", ".join(topics) + "."

    def _run(self, topic_name: str, config: Optional[RunnableConfig] = None) -> str:
        """
        Loads the content of the XML file corresponding to the given topic and returns it.
        The content is formatted and may be truncated based on configuration.
        """
        file_path = self._available_kb_files.get(topic_name)
        if not file_path:
            print(f"DEBUG: Internal KB tool: Topic '{topic_name}' not found. Returning error message.")
            return f"No internal knowledge base file found for topic: '{topic_name}'. Available topics: {self.get_available_topics_for_llm()}"

        full_content_from_file = self._load_single_xml_file(file_path)

        if not full_content_from_file:
            print(f"DEBUG: Internal KB tool: No content extracted from file for topic '{topic_name}'.")
            return f"Could not extract content from internal KB file for topic: '{topic_name}'."

        content_to_return = full_content_from_file

        return content_to_return

    def _load_single_xml_file(self, file_path: str) -> str:
        """Helper to load and format content from a single XML file."""
        all_content_parts = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            file_count = 0
            for doc_elem in root.findall('document'):
                path = None
                content = ""

                path_elem = doc_elem.find('metadata/path')
                if path_elem is not None and path_elem.text:
                    # NEW: Normalize path to use forward slashes for markdown/URL compatibility
                    path = path_elem.text.strip().replace('\\', '/')

                content_elem = doc_elem.find('content')
                if content_elem is not None and content_elem.text:
                    content = content_elem.text.strip()

                if path:
                    # Use the normalized path here
                    all_content_parts.append(f"--- FILE PATH: {path} ---\n{content}\n")
                    file_count += 1
                else:
                    print(f"WARNING: <document> element in {file_path} found without a valid <path> in <metadata>. Skipping.")
            print(f"DEBUG: Extracted {file_count} document entries from {file_path}.")
            return "\n".join(all_content_parts)
        except ET.ParseError as e:
            print(f"ERROR: Parsing internal KB XML file at {file_path}: {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred loading internal KB file from {file_path}: {e}")
        return ""
