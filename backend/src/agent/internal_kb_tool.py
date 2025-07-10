import os
import xml.etree.ElementTree as ET
from typing import Optional
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
import asyncio

class InternalKnowledgeBaseTool(BaseTool):
    name: str = "internal_knowledge_base_search"
    description: str = (
        "Selects a relevant internal knowledge base file and provides its full content "
        "to the LLM based on the user's query. "
        "Input should be the file_id of the chosen knowledge base document."
        "Note: The full content of the selected file is passed to the LLM, possibly truncated."
    )
    _xml_file_path: str

    def __init__(self, xml_file_path: str = None):
        super().__init__()
        if xml_file_path is None:
            print("No XML file path provided. Using default path.")
            xml_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../public/cg_extracted.xml"
            )
        
        self._xml_file_path = xml_file_path
        print(f"InternalKnowledgeBaseTool initialized. XML file path: {self._xml_file_path}")
        
        # Validate XML file exists
        if not os.path.exists(self._xml_file_path):
            print(f"WARNING: XML knowledge base file not found at {self._xml_file_path}")
        else:
            print(f"XML knowledge base file found at {self._xml_file_path}")

    def get_index_for_llm(self) -> str:
        """Extracts and formats the index from the XML file for LLM consumption."""
        try:
            tree = ET.parse(self._xml_file_path)
            root = tree.getroot()
            
            # Find the index section (direct child of document_collection)
            index_elem = root.find('index')
            if index_elem is None:
                return "No index found in the knowledge base."
            
            formatted_entries = []
            for file_entry in index_elem.findall('file_entry'):
                file_id = file_entry.find('file_id')
                filename = file_entry.find('filename')
                file_type = file_entry.find('file_type')
                size = file_entry.find('size_bytes')
                modified = file_entry.find('modified_time')
                
                # Get LLM analysis elements
                llm_analysis = file_entry.find('llm_analysis')
                summary_elem = None
                keywords_elem = None
                content_type_elem = None
                key_entities_elem = None
                
                if llm_analysis is not None:
                    summary_elem = llm_analysis.find('summary')
                    keywords_elem = llm_analysis.find('keywords')
                    content_type_elem = llm_analysis.find('content_type')
                    key_entities_elem = llm_analysis.find('key_entities')
                
                if file_id is not None and filename is not None:
                    entry_text = f"File ID: {file_id.text}\n"
                    entry_text += f"Filename: {filename.text}\n"
                    
                    if file_type is not None and file_type.text:
                        entry_text += f"Type: {file_type.text}\n"
                    
                    if size is not None and size.text:
                        try:
                            size_mb = round(int(size.text) / (1024 * 1024), 2)
                            entry_text += f"Size: {size_mb} MB\n"
                        except:
                            entry_text += f"Size: {size.text} bytes\n"
                    
                    if modified is not None and modified.text:
                        entry_text += f"Modified: {modified.text}\n"
                    
                    if summary_elem is not None and summary_elem.text:
                        entry_text += f"Summary: {summary_elem.text}\n"
                    
                    if keywords_elem is not None and keywords_elem.text:
                        entry_text += f"Keywords: {keywords_elem.text}\n"
                    
                    if content_type_elem is not None and content_type_elem.text:
                        entry_text += f"Content Type: {content_type_elem.text}\n"
                    
                    if key_entities_elem is not None and key_entities_elem.text:
                        entry_text += f"Key Entities: {key_entities_elem.text}\n"
                    
                    formatted_entries.append(entry_text)
            
            if not formatted_entries:
                return "No valid entries found in the knowledge base index."
            
            return "\n" + "="*50 + "\n".join(formatted_entries)
            
        except ET.ParseError as e:
            print(f"ERROR: Parsing XML file: {e}")
            return "Error parsing knowledge base index."
        except Exception as e:
            print(f"ERROR: Extracting index: {e}")
            return "Error extracting knowledge base index."
        
    def get_file_content_by_id(self, file_id: str) -> str:
        """Extracts content of a specific file by its ID from the XML."""
        try:
            tree = ET.parse(self._xml_file_path)
            root = tree.getroot()  # root is <document_collection>

            # Find the <documents> section, which is a direct child of the root
            documents_section = root.find('documents')
            if documents_section is None:
                return f"Error: <documents> section not found in XML file."

            # Iterate through each <document> inside the <documents> section
            for document in documents_section.findall('document'):
                # Find the file_id, which is a direct child of <document>
                doc_file_id_elem = document.find('file_id')
                
                if doc_file_id_elem is not None and doc_file_id_elem.text == file_id:
                    # Found the matching document, now extract its content
                    
                    # Get metadata
                    metadata = document.find('metadata')
                    path = "Unknown path"
                    filename = "Unknown filename"
                    
                    if metadata is not None:
                        path_elem = metadata.find('path')
                        filename_elem = metadata.find('filename')
                        
                        if path_elem is not None and path_elem.text:
                            path = path_elem.text.strip().replace('\\', '/')
                        if filename_elem is not None and filename_elem.text:
                            filename = filename_elem.text.strip()
                    
                    # Get content
                    content_elem = document.find('content')
                    content = "No content available"
                    if content_elem is not None and content_elem.text:
                        content = content_elem.text.strip()
                    
                    # Get LLM analysis if available
                    llm_analysis = document.find('llm_analysis')
                    llm_info = ""
                    if llm_analysis is not None:
                        summary_elem = llm_analysis.find('summary')
                        keywords_elem = llm_analysis.find('keywords')
                        content_type_elem = llm_analysis.find('content_type')
                        
                        if summary_elem is not None and summary_elem.text:
                            llm_info += f"Summary: {summary_elem.text}\n"
                        if keywords_elem is not None and keywords_elem.text:
                            llm_info += f"Keywords: {keywords_elem.text}\n"
                        if content_type_elem is not None and content_type_elem.text:
                            llm_info += f"Content Type: {content_type_elem.text}\n"
                        
                        if llm_info:
                            llm_info = "\n--- LLM ANALYSIS ---\n" + llm_info
                    
                    # Get extraction timestamp
                    timestamp_elem = document.find('extraction_timestamp')
                    timestamp = ""
                    if timestamp_elem is not None and timestamp_elem.text:
                        timestamp = f"\n--- EXTRACTED ON: {timestamp_elem.text} ---\n"
                    
                    return f"--- FILE: {filename} ---\n--- PATH: {path} ---{llm_info}{timestamp}\n{content}\n"
            
            # If the loop finishes without finding the ID
            return f"No document found with file ID: {file_id}"
            
        except ET.ParseError as e:
            print(f"ERROR: Parsing XML file: {e}")
            return f"Error parsing knowledge base for file ID: {file_id}"
        except Exception as e:
            print(f"ERROR: Extracting content for file ID {file_id}: {e}")
            return f"Error extracting content for file ID: {file_id}"

    async def _arun(self, file_id: str, config: Optional[RunnableConfig] = None) -> str:
        """Async version of the tool."""
        return await asyncio.to_thread(self._run, file_id, config)

    def _run(self, file_id: str, config: Optional[RunnableConfig] = None) -> str:
        """
        Loads the content of the document with the given file_id and returns it.
        """
        print(f"DEBUG: Internal KB tool: Retrieving content for file ID: {file_id}")
        
        # Validate input
        if not file_id or not file_id.strip():
            return "Error: Empty file_id provided."
        
        content = self.get_file_content_by_id(file_id.strip())
        
        if content.startswith("No document found") or content.startswith("Error"):
            print(f"DEBUG: Internal KB tool: {content}")
            return content
        
        print(f"DEBUG: Internal KB tool: Successfully retrieved content for file ID: {file_id}")
        return content

    def list_all_files(self) -> str:
        """Helper method to list all available files in the knowledge base."""
        try:
            tree = ET.parse(self._xml_file_path)
            root = tree.getroot()
            
            index_elem = root.find('index')
            if index_elem is None:
                return "No index found in the knowledge base."
            
            files = []
            for file_entry in index_elem.findall('file_entry'):
                file_id_elem = file_entry.find('file_id')
                filename_elem = file_entry.find('filename')
                
                if file_id_elem is not None and filename_elem is not None:
                    files.append(f"ID: {file_id_elem.text} | File: {filename_elem.text}")
            
            return "\n".join(files) if files else "No files found in index."
            
        except Exception as e:
            return f"Error listing files: {e}"