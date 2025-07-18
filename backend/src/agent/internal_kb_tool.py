import os
import xml.etree.ElementTree as ET
import logging
import time
from typing import Generator, List, Dict, Optional, Tuple, Any
from pathlib import Path
from functools import lru_cache, wraps
from dataclasses import dataclass
from enum import Enum

from agent.activity import RetrievalProgress, create_activity

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_performance(func):
    """Decorator to log function execution time and entry/exit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        logger.debug(f"ENTER {func_name} with args: {len(args)} args, {len(kwargs)} kwargs")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"EXIT {func_name} - SUCCESS - Duration: {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"EXIT {func_name} - FAILED - Duration: {execution_time:.3f}s - Error: {e}")
            raise
    return wrapper

class ContentType(Enum):
    """Enumeration of supported content types."""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"

@dataclass
class FileEntry:
    """Represents a file entry in the knowledge base index."""
    path: str
    content_type: ContentType
    keywords: List[str]
    summary: str
    file_size: Optional[int] = None
    last_modified: Optional[str] = None

@dataclass
class RetrievalResult:
    """Represents the result of a content retrieval operation."""
    success: bool
    content: str
    errors: List[str]
    retrieved_paths: List[str]
    failed_paths: List[str]

class SurgicalKBTool:
    """
    A sophisticated knowledge base tool that provides surgical retrieval capabilities.
    
    Features:
    - Efficient XML parsing with caching
    - Robust error handling and recovery
    - Flexible file path resolution
    - Memory-efficient operations
    - Comprehensive logging
    """
    
    @log_performance
    def __init__(self, kb_file_path: Optional[str] = None, max_content_size: int = 2400000):
        """
        Initialize the SurgicalKBTool.
        
        Args:
            kb_file_path: Path to the knowledge base XML file. If None, uses default path.
            max_content_size: Maximum size of content to retrieve (in bytes) for safety.
        """
        logger.info(f"Initializing SurgicalKBTool with max_content_size={max_content_size}")
        
        self.max_content_size = max_content_size
        self._xml_root: Optional[ET.Element] = None
        self._file_entries: Dict[str, FileEntry] = {}
        self._kb_file_path = self._resolve_kb_file_path(kb_file_path)
        self._is_loaded = False
        
        logger.info(f"KB file path resolved to: {self._kb_file_path}")
        logger.debug(f"Initial state: _is_loaded={self._is_loaded}, _file_entries={len(self._file_entries)}")
        
        self._load_and_prepare_kb()
        
        logger.info(f"SurgicalKBTool initialization complete. Loaded: {self._is_loaded}")

    @log_performance
    def _resolve_kb_file_path(self, kb_file_path: Optional[str]) -> str:
        """
        Resolve the knowledge base file path with multiple fallback options.
        
        Args:
            kb_file_path: Provided path or None for default.
            
        Returns:
            Resolved absolute path to the KB file.
        """
        logger.debug(f"Resolving KB file path. Input: {kb_file_path}")
        
        if kb_file_path:
            if os.path.isabs(kb_file_path):
                logger.debug(f"Using provided absolute path: {kb_file_path}")
                return kb_file_path
            else:
                logger.debug(f"Converting relative path to absolute: {kb_file_path}")
                return os.path.abspath(kb_file_path)
        
        # Default path resolution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(current_dir, "../../public/cg_extracted.xml")
        resolved_path = os.path.abspath(default_path)
        
        logger.debug(f"Using default path: {resolved_path}")
        logger.debug(f"Current directory: {current_dir}")
        
        return resolved_path

    @log_performance
    def _load_and_prepare_kb(self) -> bool:
        """
        Load and parse the XML knowledge base file.
        
        Returns:
            True if successful, False otherwise.
        """
        logger.info(f"Loading knowledge base from: {self._kb_file_path}")
        
        if not os.path.exists(self._kb_file_path):
            logger.error(f"Knowledge base file not found at {self._kb_file_path}")
            logger.debug(f"Current working directory: {os.getcwd()}")
            logger.debug(f"Files in parent directory: {os.listdir(os.path.dirname(self._kb_file_path)) if os.path.exists(os.path.dirname(self._kb_file_path)) else 'Directory not found'}")
            return False
        
        # Log file information
        try:
            file_stat = os.stat(self._kb_file_path)
            logger.info(f"KB file stats - Size: {file_stat.st_size} bytes, Modified: {time.ctime(file_stat.st_mtime)}")
        except Exception as e:
            logger.warning(f"Could not get file stats: {e}")
        
        try:
            logger.debug("Starting XML parsing...")
            # Parse XML with error handling
            tree = ET.parse(self._kb_file_path)
            self._xml_root = tree.getroot()
            logger.info(f"XML parsed successfully. Root tag: {self._xml_root.tag}")
            
            # Log XML structure information
            self._log_xml_structure()
            
            # Validate XML structure
            logger.debug("Validating XML structure...")
            if not self._validate_xml_structure():
                logger.error("Invalid XML structure in knowledge base file")
                return False
            logger.info("XML structure validation passed")
            
            # Load index entries
            logger.debug("Loading index entries...")
            self._load_index_entries()
            
            self._is_loaded = True
            logger.info(f"Knowledge base loaded successfully with {len(self._file_entries)} documents")
            
            # Log content type distribution
            self._log_content_type_distribution()
            
            return True
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.error(f"Error line: {e.lineno if hasattr(e, 'lineno') else 'unknown'}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading KB: {e}", exc_info=True)
            return False

    def _log_xml_structure(self):
        """Log information about XML structure for debugging."""
        if self._xml_root is None:
            return
        
        logger.debug(f"XML root attributes: {self._xml_root.attrib}")
        
        # Log immediate children
        children = list(self._xml_root)
        logger.debug(f"XML root has {len(children)} immediate children")
        
        for child in children:
            child_count = len(list(child))
            logger.debug(f"  - {child.tag}: {child_count} children, attributes: {child.attrib}")

    def _log_content_type_distribution(self):
        """Log the distribution of content types in the knowledge base."""
        if not self._file_entries:
            return
        
        content_type_counts = {}
        for entry in self._file_entries.values():
            content_type = entry.content_type.value
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        logger.info("Content type distribution:")
        for content_type, count in content_type_counts.items():
            logger.info(f"  - {content_type}: {count} files")

    @log_performance
    def _validate_xml_structure(self) -> bool:
        """
        Validate that the XML has the expected structure.
        
        Returns:
            True if structure is valid, False otherwise.
        """
        if self._xml_root is None:
            logger.error("XML root is None - cannot validate structure")
            return False
        
        # Check for required sections (flexible - can have either structure)
        has_index = self._xml_root.find('index') is not None
        has_documents = self._xml_root.find('documents') is not None
        has_document_collection = self._xml_root.find('document_collection') is not None
        has_content_store = self._xml_root.find('content_store') is not None
        
        logger.debug(f"XML structure check - index: {has_index}, documents: {has_documents}, "
                    f"document_collection: {has_document_collection}, content_store: {has_content_store}")
        
        valid = (has_index and has_documents) or (has_document_collection and has_content_store)
        
        if not valid:
            logger.error("XML structure validation failed - missing required sections")
            available_sections = [child.tag for child in self._xml_root]
            logger.error(f"Available sections: {available_sections}")
        
        return valid

    @log_performance
    def _load_index_entries(self) -> None:
        """Load and cache file entries from the index section."""
        logger.debug("Starting to load index entries")
        self._file_entries.clear()
        
        # Try multiple possible index structures
        index_section = (
            self._xml_root.find('index') or 
            self._xml_root.find('document_collection')
        )
        
        if index_section is None:
            logger.warning("No index section found in XML")
            return
        
        logger.debug(f"Found index section: {index_section.tag}")
        
        # Find all file entries
        file_entries = index_section.findall('.//file_entry')
        logger.info(f"Found {len(file_entries)} file entries to process")
        
        successful_entries = 0
        failed_entries = 0
        
        for i, entry in enumerate(file_entries):
            try:
                logger.debug(f"Processing file entry {i+1}/{len(file_entries)}")
                file_entry = self._parse_file_entry(entry)
                if file_entry:
                    self._file_entries[file_entry.path] = file_entry
                    successful_entries += 1
                    logger.debug(f"Successfully loaded entry: {file_entry.path}")
                else:
                    failed_entries += 1
                    logger.warning(f"Failed to parse file entry {i+1}")
            except Exception as e:
                failed_entries += 1
                logger.error(f"Exception parsing file entry {i+1}: {e}")
        
        logger.info(f"Index loading complete - Success: {successful_entries}, Failed: {failed_entries}")

    def _parse_file_entry(self, entry: ET.Element) -> Optional[FileEntry]:
        """
        Parse a file entry XML element into a FileEntry object.
        
        Args:
            entry: XML element representing a file entry.
            
        Returns:
            FileEntry object or None if parsing fails.
        """
        try:
            path = entry.findtext('path')
            if not path:
                logger.warning("File entry missing path - skipping")
                return None
            
            logger.debug(f"Parsing entry for path: {path}")
            
            # Parse content type
            content_type_str = entry.findtext('.//content_type', 'unknown').lower()
            try:
                content_type = ContentType(content_type_str)
                logger.debug(f"Content type for {path}: {content_type.value}")
            except ValueError:
                logger.warning(f"Unknown content type '{content_type_str}' for {path}, using UNKNOWN")
                content_type = ContentType.UNKNOWN
            
            # Parse keywords
            keywords_str = entry.findtext('.//keywords', '')
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            logger.debug(f"Keywords for {path}: {keywords}")
            
            summary = entry.findtext('.//summary', 'No summary available.')
            logger.debug(f"Summary for {path}: {summary[:100]}{'...' if len(summary) > 100 else ''}")
            
            # Optional metadata
            file_size_str = entry.findtext('.//file_size')
            file_size = int(file_size_str) if file_size_str and file_size_str.isdigit() else None
            if file_size:
                logger.debug(f"File size for {path}: {file_size} bytes")
            
            last_modified = entry.findtext('.//last_modified')
            if last_modified:
                logger.debug(f"Last modified for {path}: {last_modified}")
            
            return FileEntry(
                path=path,
                content_type=content_type,
                keywords=keywords,
                summary=summary,
                file_size=file_size,
                last_modified=last_modified
            )
            
        except Exception as e:
            logger.error(f"Error parsing file entry: {e}", exc_info=True)
            return None

    @lru_cache(maxsize=1)
    @log_performance
    def get_index_for_llm(self) -> str:
        """
        Returns a formatted text representation of the document index for LLM consumption.
        
        Returns:
            Formatted index text or error message.
        """
        logger.debug("Generating index for LLM consumption")
        
        if not self._is_loaded:
            logger.warning("Knowledge base index requested but not loaded")
            return "Knowledge base index is not available or failed to load."
        
        if not self._file_entries:
            logger.warning("Knowledge base index requested but no entries found")
            return "No documents found in the knowledge base index."
        
        logger.info(f"Generating index for {len(self._file_entries)} entries")
        
        index_parts = []
        for path, entry in self._file_entries.items():
            keywords_str = ', '.join(entry.keywords) if entry.keywords else 'None'
            
            entry_text = (
                f"File Path: {path}\n"
                f"  Content Type: {entry.content_type.value}\n"
                f"  Keywords: {keywords_str}\n"
                f"  Summary: {entry.summary}\n"
            )
            
            # Add optional metadata if available
            if entry.file_size:
                entry_text += f"  File Size: {entry.file_size} bytes\n"
            if entry.last_modified:
                entry_text += f"  Last Modified: {entry.last_modified}\n"
            
            entry_text += "---"
            index_parts.append(entry_text)
        
        result = "\n".join(index_parts)
        logger.debug(f"Generated index text length: {len(result)} characters")
        return result
    
    @property
    def retrieved_paths(self) -> List[str]:
        """Get the list of successfully retrieved paths from the last stream operation."""
        return getattr(self, '_retrieved_paths', [])

    @property
    def failed_paths(self) -> List[str]:
        """Get the list of failed paths from the last stream operation."""
        return getattr(self, '_failed_paths', [])

    @property
    def retrieval_errors(self) -> List[str]:
        """Get the list of errors from the last stream operation."""
        return getattr(self, '_errors', [])
    
    def stream_content_by_paths(self, paths: List[str]) -> Generator[RetrievalProgress, None, None]:
        """
        Streams content retrieval progress, yielding an update for each processed file.
        This generator is designed to power real-time progress bars and includes search statistics.
        
        Args:
            paths: List of file paths to retrieve content for.
            
        Yields:
            RetrievalProgress objects with activity updates, content chunks, and search statistics.
        """
        logger.info(f"Starting to stream content for {len(paths)} paths.")
        
        # Initialize statistics tracking as instance variables for external access
        self._retrieved_paths = []
        self._failed_paths = []
        self._errors = []
        
        if not self._is_loaded or not paths:
            status = "error" if not self._is_loaded else "completed"
            details = "Knowledge base not loaded." if not self._is_loaded else "No paths provided."
            if not self._is_loaded:
                self._errors.append("Knowledge base not loaded.")
            yield RetrievalProgress(
                activity=create_activity(
                    "retrieve_content", "Research", "Content Retrieval", details, status, "error"
                )
            )
            return # Stop the generator

        documents_section = self._xml_root.find('documents')
        if documents_section is None:
            error_msg = "XML Error: Could not find <documents> section."
            self._errors.append(error_msg)
            yield RetrievalProgress(
                activity=create_activity(
                    "retrieve_content", "Research", "Content Retrieval Failed",
                    error_msg, "error", "error"
                )
            )
            return

        # 1. Yield the initial "in_progress" state
        initial_activity = create_activity(
            id="retrieve_content", phase="Research", title=f"Reading {len(paths)} documents...",
            details="Preparing to fetch file contents.", status="in_progress", icon="read",
            progress={"current": 0, "total": len(paths)}
        )
        yield RetrievalProgress(
            activity=initial_activity
        )

        total_content_size = 0

        for i, path in enumerate(paths):
            content_chunk = None
            try:
                content = self._retrieve_single_file_content(documents_section, path)
                if content:
                    content_size = len(content.encode('utf-8'))
                    if total_content_size + content_size > self.max_content_size:
                        logger.warning(f"Content size limit exceeded. Truncating results at {path}.")
                        break # Stop processing more files
                    
                    total_content_size += content_size
                    self._retrieved_paths.append(path)
                    content_chunk = f"--- FILE PATH: {path} ---\n{content.strip()}\n"
                else:
                    # Content was empty or None
                    self._failed_paths.append(path)
                    self._errors.append(f"No content found for path: {path}")
                
                # 2. Yield a progress update after each file
                progress_activity = create_activity(
                    id="retrieve_content", phase="Research", title=f"Reading documents...",
                    details=f"Processed: {os.path.basename(path)}", status="in_progress", icon="read",
                    progress={"current": i + 1, "total": len(paths)}
                )
                yield RetrievalProgress(
                    activity=progress_activity, 
                    content_chunk=content_chunk
                )

            except Exception as e:
                error_msg = f"Error streaming content for {path}: {e}"
                logger.error(error_msg, exc_info=True)
                self._failed_paths.append(path)
                self._errors.append(error_msg)
                
                # Yield error update
                error_activity = create_activity(
                    id="retrieve_content", phase="Research", title=f"Reading documents...",
                    details=f"Error processing: {os.path.basename(path)}", status="in_progress", icon="read",
                    progress={"current": i + 1, "total": len(paths)}
                )
                yield RetrievalProgress(
                    activity=error_activity,
                    content_chunk=None
                )

        # 3. Yield the final "completed" state with comprehensive statistics
        success_count = len(self._retrieved_paths)
        failed_count = len(self._failed_paths)
        error_count = len(self._errors)
        
        final_details = f"Successfully read content for {success_count} out of {len(paths)} files."
        if failed_count > 0:
            final_details += f" Failed: {failed_count}."
        if error_count > 0:
            final_details += f" Errors: {error_count}."
        
        final_status = "completed" if success_count > 0 else "error"
        final_activity = create_activity(
            id="retrieve_content", phase="Research", title="Finished reading documents",
            details=final_details, status=final_status, icon="done",
            progress={"current": len(paths), "total": len(paths)}
        )
        
        yield RetrievalProgress(
            activity=final_activity
        )

    @log_performance
    def retrieve_by_paths(self, paths: List[str]) -> str:
        """
        Retrieve content by paths and access statistics.
        """
        content_parts = []

        logger.info(f"Retrieving content for {len(paths)} paths")
        logger.debug(f"Requested paths: {paths}")

        # Consume the generator to get all content
        for progress in self.stream_content_by_paths(paths):
            if progress.content_chunk:
                content_parts.append(progress.content_chunk)

        
        logger.info(f"Content retrieval complete - Success: {len(self.retrieved_paths)}, "
                   f"Failed: {len(self.failed_paths)}, Errors: {len(self.errors)}")

        return "\n".join(content_parts)

    @log_performance
    def retrieve_content_by_paths_detailed(self, paths: List[str]) -> RetrievalResult:
        """
        Retrieve content with detailed result information.
        
        Args:
            paths: List of file paths to retrieve content for.
            
        Returns:
            RetrievalResult with detailed information about the operation.
        """
        logger.info(f"Starting detailed content retrieval for {len(paths)} paths")
        
        if not self._is_loaded:
            error_msg = "Knowledge base not loaded"
            logger.error(error_msg)
            return RetrievalResult(
                success=False,
                content=error_msg,
                errors=[error_msg],
                retrieved_paths=[],
                failed_paths=paths
            )
        
        if not paths:
            error_msg = "No file paths provided"
            logger.warning(error_msg)
            return RetrievalResult(
                success=False,
                content="No file paths provided.",
                errors=[error_msg],
                retrieved_paths=[],
                failed_paths=[]
            )
        
        content_parts = []
        errors = []
        retrieved_paths = []
        failed_paths = []
        
        # Try multiple possible document sections
        documents_section = (
            self._xml_root.find('documents') or 
            self._xml_root.find('content_store')
        )
        
        if documents_section is None:
            error_msg = "XML Error: Could not find documents section"
            logger.error(error_msg)
            return RetrievalResult(
                success=False,
                content=error_msg,
                errors=[error_msg],
                retrieved_paths=[],
                failed_paths=paths
            )
        
        logger.debug(f"Using documents section: {documents_section.tag}")
        
        total_content_size = 0
        logger.debug(f"Maximum content size limit: {self.max_content_size} bytes")
        
        for i, path in enumerate(paths):
            logger.debug(f"Processing path {i+1}/{len(paths)}: {path}")
            
            try:
                content = self._retrieve_single_file_content(documents_section, path)
                
                if content:
                    # Check content size limits
                    content_size = len(content.encode('utf-8'))
                    logger.debug(f"Content size for {path}: {content_size} bytes")
                    
                    if total_content_size + content_size > self.max_content_size:
                        warning_msg = f"Content size limit exceeded. Truncating results at {path}"
                        logger.warning(warning_msg)
                        logger.warning(f"Total size would be: {total_content_size + content_size} bytes")
                        errors.append(warning_msg)
                        break
                    
                    total_content_size += content_size
                    content_parts.append(f"--- FILE PATH: {path} ---\n{content.strip()}\n")
                    retrieved_paths.append(path)
                    logger.debug(f"Successfully retrieved content for {path}")
                else:
                    error_msg = f"Content not found for path: {path}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    failed_paths.append(path)
                    content_parts.append(f"--- WARNING: Could not retrieve content for path: {path} ---\n")
                    
            except Exception as e:
                error_msg = f"Error retrieving content for {path}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                failed_paths.append(path)
                content_parts.append(f"--- ERROR: Failed to retrieve content for path: {path} ---\n")
        
        success = len(retrieved_paths) > 0
        content = "\n".join(content_parts) if content_parts else "No content retrieved."
        
        logger.info(f"Content retrieval summary - Total size: {total_content_size} bytes, "
                   f"Success: {success}, Retrieved: {len(retrieved_paths)}, Failed: {len(failed_paths)}")
        
        return RetrievalResult(
            success=success,
            content=content,
            errors=errors,
            retrieved_paths=retrieved_paths,
            failed_paths=failed_paths
        )


    def _retrieve_single_file_content(self, documents_section: ET.Element, path: str) -> Optional[str]:
        """
        Retrieve content for a single file path using a robust, programmatic search
        to avoid XPath injection issues with special characters.
        
        Args:
            documents_section: XML element containing documents.
            path: File path to retrieve.
            
        Returns:
            File content or None if not found.
        """

        target_path = path.replace('\\', '/')

        all_documents = documents_section.findall('document')

        # Loop through the documents in Python
        for doc_element in all_documents:
            # Find the path element within the current document's metadata
            path_element = doc_element.find('metadata/path')
            if path_element is not None and path_element.text is not None:
                xml_path = path_element.text.replace('\\', '/')
                if xml_path == target_path:
                    content = doc_element.findtext('content')
                    if content:
                        return content
        
        return None

    @log_performance
    def get_available_paths(self) -> List[str]:
        """
        Get a list of all available file paths in the knowledge base.
        
        Returns:
            List of file paths.
        """
        paths = list(self._file_entries.keys())
        logger.info(f"Retrieved {len(paths)} available paths")
        logger.debug(f"Available paths: {paths[:5]}{'...' if len(paths) > 5 else ''}")
        return paths

    def get_file_entry(self, path: str) -> Optional[FileEntry]:
        """
        Get file entry metadata for a specific path.
        
        Args:
            path: File path to look up.
            
        Returns:
            FileEntry object or None if not found.
        """
        logger.debug(f"Looking up file entry for path: {path}")
        entry = self._file_entries.get(path)
        if entry:
            logger.debug(f"Found file entry for {path}: {entry.content_type.value}")
        else:
            logger.debug(f"No file entry found for {path}")
        return entry

    @log_performance
    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """
        Search for files by keywords.
        
        Args:
            keywords: List of keywords to search for.
            
        Returns:
            List of matching file paths.
        """
        logger.info(f"Searching for files with keywords: {keywords}")
        
        if not keywords:
            logger.warning("Empty keywords list provided")
            return []
        
        keywords_lower = [k.lower() for k in keywords]
        matching_paths = []
        
        for path, entry in self._file_entries.items():
            entry_keywords_lower = [k.lower() for k in entry.keywords]
            if any(keyword in entry_keywords_lower for keyword in keywords_lower):
                matching_paths.append(path)
                logger.debug(f"Path {path} matches keywords")
        
        logger.info(f"Found {len(matching_paths)} files matching keywords")
        return matching_paths

    @log_performance
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing KB statistics.
        """
        logger.debug("Generating knowledge base statistics")
        
        if not self._is_loaded:
            logger.warning("Statistics requested but KB not loaded")
            return {"loaded": False, "error": "Knowledge base not loaded"}
        
        content_type_counts = {}
        total_size = 0
        
        for entry in self._file_entries.values():
            content_type = entry.content_type.value
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            if entry.file_size:
                total_size += entry.file_size
        
        stats = {
            "loaded": True,
            "total_documents": len(self._file_entries),
            "content_types": content_type_counts,
            "total_size_bytes": total_size if total_size > 0 else "unknown",
            "kb_file_path": self._kb_file_path
        }
        
        logger.info(f"Generated stats: {stats}")
        return stats

    @log_performance
    def reload(self) -> bool:
        """
        Reload the knowledge base from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        logger.info("Reloading knowledge base...")
        logger.debug(f"Current state before reload - Loaded: {self._is_loaded}, Entries: {len(self._file_entries)}")
        
        self._is_loaded = False
        self._file_entries.clear()
        self._xml_root = None
        
        # Clear the LRU cache
        self.get_index_for_llm.cache_clear()
        logger.debug("Cleared caches and reset state")
        
        success = self._load_and_prepare_kb()
        
        if success:
            logger.info("Knowledge base reloaded successfully")
        else:
            logger.error("Failed to reload knowledge base")
        
        return success

    def is_loaded(self) -> bool:
        """
        Check if the knowledge base is successfully loaded.
        
        Returns:
            True if loaded, False otherwise.
        """
        logger.debug(f"KB loaded status check: {self._is_loaded}")
        return self._is_loaded