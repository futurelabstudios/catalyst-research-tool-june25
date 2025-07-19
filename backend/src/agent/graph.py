import asyncio
import inspect
import os
import logging
import time
from typing import List, Dict
from functools import wraps

from agent.activity import create_activity
from agent.tools_and_schemas import SearchQueryList, Reflection, RelevantFileList
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from langchain_core.tools import tool

from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    research_instructions,
    reflection_instructions,
    answer_instructions,
    index_search_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.internal_kb_tool import SurgicalKBTool

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)

# Create specialized loggers
main_logger = logging.getLogger('research_agent.main')
search_logger = logging.getLogger('research_agent.search')
kb_logger = logging.getLogger('research_agent.kb')
performance_logger = logging.getLogger('research_agent.performance')
state_logger = logging.getLogger('research_agent.state')

def debug_configuration(config: RunnableConfig, node_name: str = ""):
    """Debug function to check configuration values"""
    try:
        configurable = Configuration.from_runnable_config(config)
        main_logger.info(f"[{node_name}] Configuration debug:")
        main_logger.info(f"  - query_generator_model: {configurable.query_generator_model}")
        main_logger.info(f"  - answer_model: {configurable.answer_model}")
        main_logger.info(f"  - reflection_model: {configurable.reflection_model}")
        main_logger.info(f"  - index_search_model: {configurable.index_search_model}")
        main_logger.info(f"  - use_web_search: {configurable.use_web_search}")
        main_logger.info(f"  - number_of_initial_queries: {configurable.number_of_initial_queries}")
        main_logger.info(f"  - max_research_loops: {configurable.max_research_loops}")
        return configurable
    except Exception as e:
        main_logger.error(f"[{node_name}] Configuration error: {str(e)}")
        if config:
            main_logger.error(f"[{node_name}] Raw config: {config}")
        raise

def log_execution_time(func):
    """Decorator to log execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        performance_logger.info(f"Starting execution of {func_name}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            performance_logger.info(f"Completed {func_name} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            performance_logger.error(f"Failed {func_name} after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

def log_state_changes(func):
    """Decorator to log state changes in nodes"""
    @wraps(func)
    def wrapper(state: OverallState, config: RunnableConfig = None, *args, **kwargs):
        func_name = func.__name__
        
        # Log input state (selective logging to avoid spam)
        state_logger.info(f"[{func_name}] Input state keys: {list(state.keys())}")
        if 'messages' in state:
            state_logger.info(f"[{func_name}] Number of messages: {len(state['messages'])}")
        if 'search_query' in state:
            state_logger.info(f"[{func_name}] Search queries: {state['search_query']}")
        if 'research_loop_count' in state:
            state_logger.info(f"[{func_name}] Research loop count: {state['research_loop_count']}")
        
        try:
            result = func(state, config, *args, **kwargs)
            
            # Check if result is an async generator
            if inspect.isasyncgen(result):
                state_logger.info(f"[{func_name}] Returning async generator (streaming function)")
                return result
            
            # Check if result is a regular generator
            if inspect.isgenerator(result):
                state_logger.info(f"[{func_name}] Returning generator (streaming function)")
                return result
            
            # Log output state changes for regular dict returns
            if result and hasattr(result, 'keys'):
                state_logger.info(f"[{func_name}] Output state changes: {list(result.keys())}")
                for key, value in result.items():
                    if key == 'messages':
                        state_logger.info(f"[{func_name}] Added {len(value)} messages")
                    elif key == 'sources_gathered':
                        state_logger.info(f"[{func_name}] Gathered {len(value)} sources")
                    elif key == 'web_research_result':
                        state_logger.info(f"[{func_name}] Research results: {len(value)} items")
                    elif key == 'relevant_file_paths':
                        state_logger.info(f"[{func_name}] Found {len(value)} relevant files: {value}")
                    else:
                        state_logger.debug(f"[{func_name}] {key}: {value}")
            elif result is None:
                state_logger.info(f"[{func_name}] No state changes returned")
            else:
                state_logger.info(f"[{func_name}] Returned non-dict result: {type(result)}")
            
            return result
        except Exception as e:
            state_logger.error(f"[{func_name}] Error processing state: {str(e)}")
            raise
    return wrapper

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    main_logger.error("GEMINI_API_KEY is not set in the environment or .env file")
    raise ValueError("GEMINI_API_KEY is not set in the environment or .env file")

main_logger.info("Initializing research agent components")

# Instantiate the new surgical tool
try:
    internal_kb_tool_instance = SurgicalKBTool(kb_file_path="public/cg_extracted_normalized.xml")
    kb_logger.info("Successfully initialized internal KB tool")
except Exception as e:
    kb_logger.error(f"Failed to initialize internal KB tool: {str(e)}")
    raise

try:
    genai_client = Client(api_key=gemini_api_key)
    main_logger.info("Successfully initialized Google GenAI client")
except Exception as e:
    main_logger.error(f"Failed to initialize Google GenAI client: {str(e)}")
    raise



@tool
@log_execution_time
def google_search_tool(query: str) -> dict:
    """Performs a Google search and returns a summarized result with citations."""
    search_logger.info(f"Performing Google search for query: '{query}'")
    
    try:
        configurable = Configuration.from_runnable_config(None)
        search_logger.debug(f"Using search model: {configurable.query_generator_model}")
        
        llm_for_search = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model, temperature=0, max_retries=2, api_key=gemini_api_key
        )
        
        formatted_prompt = research_instructions.format(
            current_date=get_current_date(), research_topic=query
        )
        search_logger.debug(f"Formatted search prompt length: {len(formatted_prompt)} characters")
        
        response = genai_client.models.generate_content(
            model=llm_for_search.model_name,
            contents=formatted_prompt,
            config={"tools": [{"google_search": {}}], "temperature": 0},
        )
        
        sources_gathered = []
        modified_text = response.text
        search_logger.info(f"Received search response of length: {len(modified_text)} characters")
        
        if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
            search_logger.info("Processing grounding metadata and citations")
            resolved_urls = resolve_urls(response.candidates[0].grounding_metadata.grounding_chunks, 0)
            citations = get_citations(response, resolved_urls)
            modified_text = insert_citation_markers(response.text, citations)
            sources_gathered = [item for citation in citations for item in citation["segments"]]
            search_logger.info(f"Found {len(sources_gathered)} sources with citations")
        else:
            search_logger.warning("No grounding metadata found in search response")
        
        search_logger.info(f"Successfully completed Google search for query: '{query}'")
        return {"content": modified_text, "sources": sources_gathered}
    
    except Exception as e:
        search_logger.error(f"Error in Google search for query '{query}': {str(e)}")
        raise

# =========================================================================
# NODES
# =========================================================================

@log_state_changes
@log_execution_time
def route_initial_query(state: OverallState, config: RunnableConfig) -> Dict:
    """
    Node that determines whether to use the web search path or the internal KB path.
    This is the new entry point for the graph.
    """
    main_logger.info("Routing initial query to determine search path")
    
    # Initial routing activity
    routing_activity = create_activity(
        "query_routing", "Planning", "Analyzing query type...", 
        "Determining optimal research strategy", "in_progress", "brain"
    )
    
    try:
        configurable = Configuration.from_runnable_config(config)
        use_web_search = configurable.use_web_search
        
        main_logger.info(f"Configuration determined: use_web_search = {use_web_search}")
        
        if use_web_search:
            main_logger.info("Routing to web search path")
            routing_complete = create_activity(
                "query_routing", "Planning", "Web research path selected", 
                "Query requires web search for current information", "completed", "done"
            )
            return {
                "next_node": "generate_web_queries",
                "activity_feed": [routing_activity, routing_complete]
            }
        else:
            main_logger.info("Routing to internal KB path")
            routing_complete = create_activity(
                "query_routing", "Planning", "Knowledge base path selected", 
                "Query can be answered from internal knowledge base", "completed", "done"
            )
            return {
                "next_node": "search_kb_index",
                "activity_feed": [routing_activity, routing_complete]
            }
    
    except Exception as e:
        main_logger.error(f"Error in route_initial_query: {str(e)}")
        error_activity = create_activity(
            "query_routing", "Planning", "Query routing failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        return {"activity_feed": [routing_activity, error_activity]}

# --- Web Search Path ---
@log_state_changes
@log_execution_time
def generate_web_queries(state: OverallState, config: RunnableConfig) -> Dict:
    """
    Node for the web search path. Generates initial search queries.
    """
    main_logger.info("Generating web search queries")
    
    # Activity start
    activity_start = create_activity(
        "query_generation", "Planning", "Analyzing your question...", 
        "Breaking down the research topic into targeted search queries", "in_progress", "brain"
    )
    
    try:
        configurable = Configuration.from_runnable_config(config)
        user_query = get_research_topic(state["messages"])
        
        main_logger.info(f"Research topic extracted: '{user_query}'")
        main_logger.info(f"Target number of queries: {configurable.number_of_initial_queries}")
        
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model, temperature=1.0, max_retries=2, api_key=gemini_api_key
        )
        structured_llm = llm.with_structured_output(SearchQueryList)
        
        formatted_prompt = query_writer_instructions.format(
            current_date=get_current_date(),
            research_topic=user_query,
            number_queries=configurable.number_of_initial_queries,
        )
        
        main_logger.debug(f"Generated prompt length: {len(formatted_prompt)} characters")
        
        result = structured_llm.invoke(formatted_prompt)
        
        main_logger.info(f"Generated {len(result.query)} search queries: {result.query}")
        
        # Activity end
        activity_end = create_activity(
            "query_generation", "Planning", "Research strategy prepared", 
            f"Generated {len(result.query)} focused search queries", "completed", "done"
        )
        
        return {
            "search_query": result.query,
            "activity_feed": [activity_start, activity_end]
        }
    
    except Exception as e:
        main_logger.error(f"Error in generate_web_queries: {str(e)}")
        error_activity = create_activity(
            "query_generation", "Planning", "Query generation failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        return {"activity_feed": [activity_start, error_activity]}

@log_state_changes
@log_execution_time
def perform_web_search(state: OverallState, config: RunnableConfig) -> Dict:
    """
    Node that performs the web search for a given list of queries.
    """
    queries = state.get("search_query", [])
    current_loop = state.get("research_loop_count", 0)
    
    search_logger.info(f"Starting web search iteration {current_loop + 1}")
    search_logger.info(f"Performing web search for {len(queries)} queries: {queries}")
    
    activities = []
    
    # Main batch activity
    main_activity = create_activity(
        "web_search_batch", "Research", f"Web research batch {current_loop + 1}", 
        f"Executing {len(queries)} targeted searches", "in_progress", "search",
        progress={"current": 0, "total": len(queries)}
    )
    activities.append(main_activity)
    
    try:
        gathered_contents = []
        gathered_sources = []
        
        for i, query in enumerate(queries):
            search_logger.info(f"Processing query {i+1}/{len(queries)}: '{query}'")
            
            # Individual query activity
            query_activity = create_activity(
                f"search_query_{i}", "Research", f"Searching: {query[:50]}...", 
                f"Query {i+1} of {len(queries)}", "in_progress", "search"
            )
            activities.append(query_activity)
            
            try:
                tool_output = google_search_tool.invoke({"query": query})
                gathered_contents.append(tool_output["content"])
                gathered_sources.extend(tool_output["sources"])
                
                search_logger.info(f"Query {i+1} completed. Content length: {len(tool_output['content'])}, Sources: {len(tool_output['sources'])}")
                
                # Query completion
                query_complete = create_activity(
                    f"search_query_{i}", "Research", f"Search completed", 
                    f"Found {len(tool_output['sources'])} sources", "completed", "check-circle"
                )
                activities.append(query_complete)
                
            except Exception as e:
                search_logger.error(f"Error in query {i+1}: {str(e)}")
                error_activity = create_activity(
                    f"search_query_{i}", "Research", f"Search failed", 
                    f"Error: {str(e)}", "failed", "error", retryable=True
                )
                activities.append(error_activity)

        search_logger.info(f"Web search completed. Total content items: {len(gathered_contents)}, Total sources: {len(gathered_sources)}")
        
        # Content processing activity
        processing_activity = create_activity(
            "content_processing", "Analysis", "Processing search results...", 
            f"Analyzing content from {len(gathered_contents)} sources", "in_progress", "brain"
        )
        activities.append(processing_activity)
        
        # Batch completion
        batch_complete = create_activity(
            "web_search_batch", "Research", f"Research batch {current_loop + 1} complete", 
            f"Total: {len(gathered_contents)} results, {len(gathered_sources)} sources", 
            "completed", "done"
        )
        activities.append(batch_complete)
        
        # Content processing complete
        content_complete = create_activity(
            "content_processing", "Analysis", "Content analysis complete", 
            f"Processed {len(gathered_sources)} unique sources", "completed", "done"
        )
        activities.append(content_complete)
        
        return {
            "web_research_result": gathered_contents,
            "sources_gathered": gathered_sources,
            "research_loop_count": current_loop + 1,
            "activity_feed": activities,
        }
    
    except Exception as e:
        search_logger.error(f"Error in perform_web_search: {str(e)}")
        error_activity = create_activity(
            "web_search_batch", "Research", f"Web search failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        activities.append(error_activity)
        return {"activity_feed": activities}

# --- Internal KB Path ---
@log_state_changes
@log_execution_time
def search_kb_index(state: OverallState, config: RunnableConfig) -> Dict:
    """
    Node for the internal KB path. Searches the KB index to find relevant file paths.
    If no relevant files are found, proceeds directly to answer generation using LLM knowledge.
    """
    kb_logger.info("Searching KB index for relevant files")

    activity_start = create_activity(
        "search_index", "Research", "Searching knowledge base index...", 
        "Analyzing internal knowledge base for relevant documents", "in_progress", "search"
    )
    
    try:
        configurable = debug_configuration(config, "search_kb_index")
        user_query = get_research_topic(state["messages"])
        
        kb_logger.info(f"KB search query: '{user_query}'")
        kb_logger.info(f"Using model to search index: {configurable.index_search_model}")
        
        # Index analysis activity
        index_activity = create_activity(
            "index_analysis", "Analysis", "Analyzing knowledge base structure...", 
            "Processing document index for relevance matching", "in_progress", "brain"
        )
        
        llm = ChatGoogleGenerativeAI(
            model=configurable.index_search_model, temperature=0.0, max_retries=2, api_key=gemini_api_key
        )
        structured_llm = llm.with_structured_output(RelevantFileList)

        # Get the entire index from our tool
        kb_logger.info("Retrieving KB index from tool")
        kb_index_text = internal_kb_tool_instance.get_index_for_llm()
        kb_logger.info(f"KB index retrieved. Length: {len(kb_index_text)} characters")
        
        formatted_prompt = index_search_instructions.format(
            research_topic=user_query,
            kb_index=kb_index_text,
        )
        
        kb_logger.debug(f"Formatted KB search prompt length: {len(formatted_prompt)} characters")
        
        result = structured_llm.invoke(formatted_prompt)
        
        if not result.file_paths:
            kb_logger.warning("No relevant files found in KB index - proceeding to answer from LLM knowledge")
            
            # Index analysis complete (no matches)
            index_complete = create_activity(
                "index_analysis", "Analysis", "No relevant documents found", 
                "Knowledge base contains no matching documents", "completed", "alert-circle"
            )
            
            # Fallback to LLM knowledge activity
            fallback_activity = create_activity(
                "llm_fallback", "Analysis", "Using general knowledge...", 
                "Answering from trained knowledge base", "in_progress", "brain"
            )
            
            activity_end = create_activity(
                "search_index", "Research", "Proceeding with general knowledge", 
                "No specific documents found - using LLM training data", "completed", "brain"
            )
            
            # Set empty research result to trigger direct answer generation
            # This will cause retrieve_kb_content to be skipped and go straight to finalize_answer
            return {
                "relevant_file_paths": [],  # Empty list
                "web_research_result": [],  # Empty list - will be handled in finalize_answer
                "activity_feed": [activity_start, index_activity, index_complete, fallback_activity, activity_end]
            }

        kb_logger.info(f"Found {len(result.file_paths)} relevant files: {result.file_paths}")
        
        # Index analysis complete
        index_complete = create_activity(
            "index_analysis", "Analysis", "Document matching complete", 
            f"Identified {len(result.file_paths)} relevant documents", "completed", "done"
        )
        
        activity_end = create_activity(
            "search_index", "Research", "Knowledge base search complete", 
            f"Found {len(result.file_paths)} potentially relevant files", "completed", "done"
        )
        
        return {
            "relevant_file_paths": result.file_paths, 
            "activity_feed": [activity_start, index_activity, index_complete, activity_end]
        }
    
    except Exception as e:
        kb_logger.error(f"Error in search_kb_index: {str(e)}")
        error_activity = create_activity(
            "search_index", "Research", "Knowledge base search failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        return {"activity_feed": [activity_start, error_activity]}


@log_state_changes
@log_execution_time
def retrieve_kb_content(state: OverallState, config: RunnableConfig) -> Dict:
    """
    Node that retrieves content from KB files and collects all activity updates.
    If no file paths are provided, skips content retrieval (fallback to LLM knowledge).
    """
    file_paths = state.get("relevant_file_paths", [])
    kb_logger.info(f"Retrieving content for {len(file_paths)} files.")

    if not file_paths:
        kb_logger.info("No relevant file paths provided - skipping content retrieval for LLM fallback")
        
        # Create activity to indicate we're skipping KB retrieval
        skip_activity = create_activity(
            "skip_retrieval", "Analysis", "Skipping document retrieval", 
            "No relevant documents found - will use general knowledge", "completed", "brain"
        )
        
        return {
            "web_research_result": [],  # Empty - will trigger LLM knowledge fallback
            "activity_feed": [skip_activity]
        }

    all_content_chunks = []
    all_activities = []
    
    try:
        # Get the generator from the tool
        tool_generator = internal_kb_tool_instance.stream_content_by_paths(file_paths)
        
        # Loop through the generator and collect all updates
        for progress_update in tool_generator:
            # Collect the activity update
            if progress_update.activity:
                all_activities.append(progress_update.activity)

            # Collect the content chunk
            if progress_update.content_chunk:
                all_content_chunks.append(progress_update.content_chunk)
            
            # No need for asyncio.sleep in a synchronous function

        # After the loop is done, prepare the final content
        final_content = "\n".join(all_content_chunks)
        kb_logger.info(f"Content retrieval finished. Total content length: {len(final_content)} characters.")

        # Return all collected data as a single state update
        return {
            "web_research_result": [final_content] if final_content else [],
            "activity_feed": all_activities
        }
    
    except Exception as e:
        kb_logger.error(f"Error in retrieve_kb_content: {str(e)}")
        error_activity = create_activity(
            "retrieve_content", "Research", "Content Retrieval Failed",
            str(e), "error", "error"
        )
        return {
            "web_research_result": [f"Error during retrieval: {e}"],
            "activity_feed": [error_activity]
        }


# --- Shared Nodes ---
@log_state_changes
@log_execution_time
def reflection(state: OverallState, config: RunnableConfig) -> Dict:
    """
    This node now primarily serves the web search path for iterative refinement.
    """
    main_logger.info("Starting reflection on gathered research")
    
    # Reflection start activity
    reflection_start = create_activity(
        "reflection", "Analysis", "Evaluating research quality...", 
        "Assessing information completeness and identifying gaps", "in_progress", "brain"
    )
    
    try:
        configurable = Configuration.from_runnable_config(config)
        research_results = state["web_research_result"]
        
        main_logger.info(f"Reflecting on {len(research_results)} research results")
        main_logger.info(f"Using reflection model: {configurable.reflection_model}")
        
        formatted_prompt = reflection_instructions.format(
            current_date=get_current_date(),
            research_topic=get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(research_results),
        )
        
        main_logger.debug(f"Reflection prompt length: {len(formatted_prompt)} characters")
        
        llm = ChatGoogleGenerativeAI(
            model=configurable.reflection_model, temperature=1.0, max_retries=2, api_key=gemini_api_key
        )
        result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
        
        main_logger.info(f"Reflection completed. Is sufficient: {result.is_sufficient}")
        if result.follow_up_queries:
            main_logger.info(f"Generated {len(result.follow_up_queries)} follow-up queries: {result.follow_up_queries}")
        
        # Reflection completion activity
        if result.is_sufficient:
            reflection_end = create_activity(
                "reflection", "Analysis", "Research evaluation complete", 
                "Information appears sufficient for comprehensive answer", "completed", "check-circle"
            )
        else:
            reflection_end = create_activity(
                "reflection", "Analysis", "Additional research needed", 
                f"Identified {len(result.follow_up_queries)} areas for deeper investigation", "completed", "search"
            )
        
        return {
            "is_sufficient": result.is_sufficient,
            "search_query": result.follow_up_queries,
            "activity_feed": [reflection_start, reflection_end]
        }
    
    except Exception as e:
        main_logger.error(f"Error in reflection: {str(e)}")
        error_activity = create_activity(
            "reflection", "Analysis", "Research evaluation failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        return {"activity_feed": [reflection_start, error_activity]}

@log_execution_time
def evaluate_research(state: OverallState, config: RunnableConfig) -> str:
    """
    Routing function. For internal KB, it always finalizes. For web search, it iterates.
    """
    main_logger.info("Evaluating research completeness")
    
    try:
        configurable = Configuration.from_runnable_config(config)
        
        # If we used the internal KB, we go straight to the answer.
        if not configurable.use_web_search:
            main_logger.info("Internal KB path - proceeding to finalize answer")
            return "finalize_answer"

        # For web search, check for iteration
        max_loops = configurable.max_research_loops
        current_loop = state.get("research_loop_count", 0)
        is_sufficient = state.get("is_sufficient", False)
        
        main_logger.info(f"Web search evaluation - Loop: {current_loop}/{max_loops}, Sufficient: {is_sufficient}")
        
        if is_sufficient or current_loop >= max_loops:
            main_logger.info("Research complete - proceeding to finalize answer")
            return "finalize_answer"
        else:
            main_logger.info("Research needs more iteration - continuing web search")
            return "perform_web_search"
    
    except Exception as e:
        main_logger.error(f"Error in evaluate_research: {str(e)}")
        raise

@log_state_changes
@log_execution_time
def finalize_answer(state: OverallState, config: RunnableConfig) -> Dict:
    """
    This node generates the final answer based on the content gathered in the 
    'web_research_result' field. If no research content is available, it answers
    from the LLM's own knowledge.
    """
    main_logger.info("Finalizing answer based on gathered research")
    
    activities = []
    
    # Synthesis start activity
    synthesis_start = create_activity(
        "synthesize", "Synthesis", "Composing final answer...", 
        "Preparing comprehensive response", "in_progress", "brain"
    )
    activities.append(synthesis_start)
    
    try:
        configurable = debug_configuration(config, "finalize_answer")
        reasoning_model = configurable.answer_model
        current_date = get_current_date()
        research_results = state.get("web_research_result", [])
        
        main_logger.info(f"Using model to finalize answer: {reasoning_model}")
        main_logger.info(f"Finalizing answer based on {len(research_results)} research results")
        
        # Check if we have research content or need to use LLM knowledge
        has_research_content = bool(research_results and any(result.strip() for result in research_results))
        
        if has_research_content:
            # Information synthesis activity (with research content)
            info_synthesis = create_activity(
                "information_synthesis", "Synthesis", "Synthesizing research information...", 
                f"Combining insights from {len(research_results)} sources", "in_progress", "brain"
            )
            activities.append(info_synthesis)
            
            formatted_prompt = answer_instructions.format(
                current_date=current_date,
                research_topic=get_research_topic(state["messages"]),
                summaries="\n---\n\n".join(research_results),
            )
            
            main_logger.info("Using research-based answer generation")
            
        else:
            # LLM knowledge synthesis activity (fallback)
            info_synthesis = create_activity(
                "llm_knowledge_synthesis", "Synthesis", "Drawing from general knowledge...", 
                "Generating answer from trained knowledge base", "in_progress", "brain"
            )
            activities.append(info_synthesis)
            
            # Modified prompt for LLM-only knowledge
            user_query = get_research_topic(state["messages"])
            formatted_prompt = f"""You are a knowledgeable research assistant. Please provide a comprehensive and accurate answer to the following question based on your training data and general knowledge.

Current date: {current_date}
Research topic: {user_query}

Please provide a detailed, well-structured answer that addresses the user's question thoroughly. Use your general knowledge and training data to provide accurate information. If there are any limitations to your knowledge or if the information might be outdated, please mention this appropriately.

Answer:"""
            
            main_logger.info("Using LLM knowledge fallback for answer generation")
        
        main_logger.debug(f"Answer prompt length: {len(formatted_prompt)} characters")
        
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model, temperature=0, max_retries=2, api_key=gemini_api_key
        )
        result = llm.invoke(formatted_prompt)
        
        main_logger.info(f"Generated answer length: {len(result.content)} characters")
        
        # Information synthesis complete
        if has_research_content:
            info_complete = create_activity(
                "information_synthesis", "Synthesis", "Research synthesis complete", 
                f"Successfully combined insights from research sources", "completed", "done"
            )
        else:
            info_complete = create_activity(
                "llm_knowledge_synthesis", "Synthesis", "Knowledge synthesis complete", 
                f"Answer generated from general knowledge", "completed", "done"
            )
        activities.append(info_complete)
        
        unique_sources = []
        if configurable.use_web_search and state.get("sources_gathered"):
            # Citation processing activity
            citation_activity = create_activity(
                "citation_processing", "Synthesis", "Processing citations...", 
                f"Linking {len(state['sources_gathered'])} sources", "in_progress", "web"
            )
            activities.append(citation_activity)
            
            original_sources = len(state["sources_gathered"])
            main_logger.info(f"Processing {original_sources} sources for citation replacement")
            
            for source in state["sources_gathered"]:
                if "short_url" in source and source["short_url"] in result.content:
                    result.content = result.content.replace(source["short_url"], source["value"])
                    unique_sources.append(source)
            
            main_logger.info(f"Processed citations: {len(unique_sources)} unique sources used")
            
            # Citation processing complete
            citation_complete = create_activity(
                "citation_processing", "Synthesis", "Citations processed", 
                f"Successfully linked {len(unique_sources)} unique sources", "completed", "done"
            )
            activities.append(citation_complete)
        
        # Quality validation activity
        validation_activity = create_activity(
            "quality_check", "Validation", "Validating answer quality...", 
            "Ensuring completeness and accuracy", "in_progress", "check-circle"
        )
        activities.append(validation_activity)
        
        # Quality validation complete
        validation_complete = create_activity(
            "quality_check", "Validation", "Quality validation complete", 
            "Answer meets quality standards", "completed", "check-circle"
        )
        activities.append(validation_complete)
        
        main_logger.info("Answer finalization completed successfully")
        
        # Final synthesis activity
        answer_source = "research-based" if has_research_content else "knowledge-based"
        synthesis_end = create_activity(
            "synthesize", "Synthesis", "Answer composed", 
            f"Final {answer_source} answer prepared ({len(result.content)} characters)", "completed", "done"
        )
        activities.append(synthesis_end)
        
        # Combine with existing activities from state
        all_activities = state.get("activity_feed", []) + activities
        
        return {
            "messages": [AIMessage(content=result.content)], 
            "sources_gathered": unique_sources, 
            "activity_feed": all_activities
        }
    
    except Exception as e:
        main_logger.error(f"Error in finalize_answer: {str(e)}")
        error_activity = create_activity(
            "synthesize", "Synthesis", "Answer composition failed", 
            f"Error: {str(e)}", "failed", "error"
        )
        activities.append(error_activity)
        return {"activity_feed": activities}

# =========================================================================
# GRAPH CONSTRUCTION
# =========================================================================

main_logger.info("Constructing research agent graph")

# Define the new graph structure
builder = StateGraph(OverallState, config_schema=Configuration)

# Add nodes
builder.add_node("route_initial_query", route_initial_query)
builder.add_node("generate_web_queries", generate_web_queries)
builder.add_node("perform_web_search", perform_web_search)
builder.add_node("search_kb_index", search_kb_index)
builder.add_node("retrieve_kb_content", retrieve_kb_content)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

main_logger.info("Added all nodes to graph")

# Define the edges
builder.set_entry_point("route_initial_query")

# Conditional routing from the entry point
builder.add_conditional_edges(
    "route_initial_query",
    lambda x: x["next_node"],
    {
        "generate_web_queries": "generate_web_queries",
        "search_kb_index": "search_kb_index",
    }
)

# Web Search Path
builder.add_edge("generate_web_queries", "perform_web_search")
builder.add_edge("perform_web_search", "reflection")

# Internal KB Path
builder.add_edge("search_kb_index", "retrieve_kb_content")
builder.add_edge("retrieve_kb_content", "finalize_answer")

# Conditional edge from reflection (for web search iteration)
builder.add_conditional_edges(
    "reflection",
    evaluate_research,
    {
        "perform_web_search": "perform_web_search",
        "finalize_answer": "finalize_answer"
    }
)

# Final step
builder.add_edge("finalize_answer", END)

main_logger.info("Graph edges configured successfully")

graph = builder.compile(name="pro-search-agent")
main_logger.info("Research agent graph compiled successfully")

# Log graph structure for debugging
main_logger.info("Graph structure:")
main_logger.info(f"Nodes: {list(graph.nodes.keys())}")
main_logger.info("Research agent initialization complete")