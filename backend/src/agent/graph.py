import os
from typing import List, AsyncGenerator
import asyncio


from agent.tools_and_schemas import SearchQueryList, Reflection, FileSelection, KBReflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig, RunnableParallel
from google.genai import Client
from langchain_core.tools import tool

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    research_instructions,
    reflection_instructions,
    answer_instructions,
    kb_file_selector_instructions,
    kb_reflection_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.internal_kb_tool import InternalKnowledgeBaseTool

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Initialize the Internal Knowledge Base Tool
internal_kb_tool_instance = InternalKnowledgeBaseTool(xml_file_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../public/cg_extracted.xml"
))

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# Wrap the Google Search logic into a LangChain tool for consistency and easier use
@tool
async def google_search_tool(query: str) -> dict:
    """Performs a Google search and returns a summarized result with citations."""
    configurable = Configuration.from_runnable_config(None) # Dummy config to get model name
    
    llm_for_search = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = research_instructions.format(
        current_date=get_current_date(),
        research_topic=query,
    )

    response = await asyncio.to_thread(
        genai_client.models.generate_content,
        model=llm_for_search.model_name,
        contents=formatted_prompt,
        config={"tools": [{"google_search": {}}], "temperature": 0},
    )

    sources_gathered = []
    modified_text = response.text
    if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
        resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, 0 # ID might need to be dynamic for real parallel execution
        )
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [item for citation in citations for item in citation["segments"]]
    
    return {"content": modified_text, "sources": sources_gathered, "input_query": query}


# Nodes
def determine_tool_and_initial_action(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    LangGraph node to determine whether to use web search or internal KB,
    and then prepare the state for the next step (either generating web queries or routing KB).
    """
    configurable = Configuration.from_runnable_config(config)
    user_query = get_research_topic(state["messages"])
    
    if configurable.use_web_search:
        print("DEBUG: Using Web Search path. Generating initial queries.")
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        structured_llm = llm.with_structured_output(SearchQueryList)

        current_date = get_current_date()
        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=user_query,
            number_queries=configurable.number_of_initial_queries,
        )
        
        result = structured_llm.invoke(formatted_prompt)
        return {
            "query_list": result.query, 
            "use_web_search": True,
            "research_loop_count": 0  # Initialize here
        }
    else:
        print("DEBUG: Using Internal Knowledge Base path. Selecting relevant files from index.")
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model,
            temperature=0.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        structured_llm = llm.with_structured_output(FileSelection)

        files_index = internal_kb_tool_instance.get_index_for_llm()
        formatted_prompt = kb_file_selector_instructions.format(
            files_index=files_index,
            user_query=user_query,
        )
        
        try:
            file_selection_result = structured_llm.invoke(formatted_prompt)
            selected_file_ids = file_selection_result.selected_file_ids
            print(f"DEBUG: LLM selected file IDs: {selected_file_ids}")
            return {
                "search_query": selected_file_ids, 
                "use_web_search": False, 
                "fetched_file_ids": selected_file_ids,
                "research_loop_count": 0  # Initialize here
            }
        except Exception as e:
            print(f"ERROR: Failed to select files from KB index: {e}")
            return {
                "web_research_result": [f"I could not select relevant files from the knowledge base. Error: {e}"],
                "messages": [AIMessage(content=f"I could not select relevant files from the knowledge base. Error: {e}")],
                "use_web_search": False,
                "fetched_file_ids": [],
                "research_loop_count": 0  # Initialize here
            }


async def research_step(state: OverallState, config: RunnableConfig) -> AsyncGenerator[OverallState, None]:
    """LangGraph node that performs research and STREAMS progress updates."""
    configurable = Configuration.from_runnable_config(config)
    use_web_search = state.get("use_web_search", configurable.use_web_search)

    # Initialize counts for the current research loop if they don't exist
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries
    if state.get("max_research_loops") is None:
        state["max_research_loops"] = configurable.max_research_loops
    
    current_loop_queries: List[str] = []
    if use_web_search:
        current_loop_queries = state.get("query_list", [])
        if not current_loop_queries:
            current_loop_queries = [get_research_topic(state["messages"])]
    else:
        if state.get("chosen_kb_topic"):
            current_loop_queries = [state["chosen_kb_topic"]]
        else:
            # Handle error case
            yield {
                "web_research_result": ["Internal KB path active but no topic was selected."],
                # Don't update research_loop_count here to avoid conflicts
            }
            return

    ran_queries = state.get("search_query", [])
    
    # Initialize lists to store results
    gathered_contents = []
    gathered_sources = []

    # Yield a "planning" message before starting
    yield {
        "messages": [
            AIMessage(
                content=f"Starting research with {len(current_loop_queries)} queries.",
                name="tool_code",
                tool_call_id="research_update",
            )
        ]
    }

    if use_web_search:
        # Use RunnableParallel to execute all google searches concurrently
        # Note: We need to make google_search_tool async
        print(f"DEBUG: Performing {len(current_loop_queries)} Web Searches in parallel...")
        parallel_searches = google_search_tool.batch_as_completed(
            [{"query": q} for q in current_loop_queries], 
            config=config
        )
        
        for search_task in parallel_searches:
            try:
                # As each search completes, yield an update
                tool_output = await search_task
                query_used = tool_output['input_query'] # We'll modify the tool to return this
                
                yield {
                    "messages": [
                        AIMessage(
                            content=f"Finished search for: '{query_used}'. Found {len(tool_output['sources'])} sources.",
                            name="tool_code",
                            tool_call_id="research_update",
                        )
                    ]
                }
                gathered_contents.append(tool_output["content"])
                gathered_sources.extend(tool_output["sources"])
                ran_queries.append(query_used)
            except Exception as e:
                yield {"messages": [AIMessage(content=f"Search failed for a query. Error: {e}", name="tool_code", tool_call_id="research_update")]}
    
    else: # Internal KB path
        for file_id in current_loop_queries:
            yield {"messages": [AIMessage(content=f"Retrieving from Internal KB: file ID '{file_id}'", name="tool_code", tool_call_id="research_update")]}
            await asyncio.sleep(0.5)
            
            # Add debug logging to see what's being retrieved
            try:
                kb_content = await internal_kb_tool_instance.ainvoke(file_id, config=config)
                print(f"DEBUG: Retrieved content for file ID {file_id}: {kb_content[:200]}...")  # First 200 chars
                
                if not kb_content or kb_content.strip() == "":
                    print(f"WARNING: Empty or no content retrieved for file ID {file_id}")
                    kb_content = f"No content found for file ID: {file_id}"
                
                gathered_contents.append(kb_content)
                ran_queries.append(file_id)
                yield {"messages": [AIMessage(content=f"Finished retrieving file ID: '{file_id}' (Content length: {len(kb_content)} chars)", name="tool_code", tool_call_id="research_update")]}
                
            except Exception as e:
                print(f"ERROR: Failed to retrieve content for file ID {file_id}: {e}")
                error_content = f"Error retrieving file ID {file_id}: {e}"
                gathered_contents.append(error_content)
                ran_queries.append(file_id)
                yield {"messages": [AIMessage(content=f"Error retrieving file ID: '{file_id}' - {e}", name="tool_code", tool_call_id="research_update")]}

    # Yield the final accumulated state for this node
    # Only increment research_loop_count in the final yield to avoid conflicts
    current_research_count = state.get("research_loop_count", 0)
    yield {
        "sources_gathered": gathered_sources,
        "search_query": ran_queries,
        "web_research_result": gathered_contents,
        "research_loop_count": current_research_count + 1,  # Only update once at the end
        "number_of_ran_queries": len(ran_queries),
    }



def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = configurable.reflection_model
    use_web_search = state.get("use_web_search", configurable.use_web_search)

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    
    if use_web_search:
        # Use existing web search reflection
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=research_topic,
            summaries="\n\n---\n\n".join(state["web_research_result"]),
        )
        
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
        
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    else:
        # Use KB-specific reflection
        files_index = internal_kb_tool_instance.get_index_for_llm()
        formatted_prompt = kb_reflection_instructions.format(
            user_query=research_topic,
            files_index=files_index,
            summaries="\n\n---\n\n".join(state["web_research_result"]),
        )
        
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        result = llm.with_structured_output(KBReflection).invoke(formatted_prompt)
        
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.suggested_file_ids,  # Use file_ids as follow-up queries
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> str:  # Changed return type - no more Send for parallel processing
    """LangGraph routing function that determines the next step in the research flow."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # Continue research but prepare queries for the next iteration
        # Store follow-up queries in state to be processed sequentially
        return "prepare_followup_research"


def prepare_followup_research(state: ReflectionState, config: RunnableConfig) -> OverallState:
    """Prepare follow-up research queries without parallel processing to avoid state conflicts."""
    configurable = Configuration.from_runnable_config(config)
    
    if configurable.use_web_search:
        # For web search, set up follow-up queries
        if not state["follow_up_queries"]:
            # No follow-up queries, mark as sufficient to trigger finalization
            return {"is_research_complete": True}
        return {
            "query_list": state["follow_up_queries"],
            "use_web_search": True,
            "is_research_complete": False,
        }
    else:
        # For internal KB, filter out already fetched file IDs
        fetched_ids = set(state.get("fetched_file_ids", []))
        new_file_ids = [fid for fid in state["follow_up_queries"] if fid not in fetched_ids]
        
        if not new_file_ids:
            print("DEBUG: Internal KB: No new file IDs to fetch. Finalizing answer.")
            return {"is_research_complete": True}  # Signal completion
        
        # Update fetched_file_ids to include new ones
        updated_fetched_ids = list(fetched_ids) + new_file_ids
        
        return {
            "search_query": new_file_ids,
            "chosen_kb_topic": None,  # Reset this
            "fetched_file_ids": updated_fetched_ids,
            "use_web_search": False,
            "is_research_complete": False,
        }


def check_research_completion(state: OverallState) -> str:
    """Check if research is complete or if we need to continue."""
    if state.get("is_research_complete", False):
        return "finalize_answer"
    else:
        return "research_step"


def finalize_answer(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that finalizes the research summary."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = configurable.answer_model # Use model from config

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    unique_sources = []
    # Only process web citations if web search was enabled AND sources were gathered
    if configurable.use_web_search and state.get("sources_gathered"):
        for source in state["sources_gathered"]:
            # Check if source['short_url'] exists before replacement for robustness
            if "short_url" in source and source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)
    else:
        # For internal KB, sources_gathered is empty, no special handling needed here.
        # The prompt handles internal KB references without formal citations.
        pass

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("determine_tool_and_initial_action", determine_tool_and_initial_action)
builder.add_node("research_step", research_step)
builder.add_node("reflection", reflection)
builder.add_node("prepare_followup_research", prepare_followup_research)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint
builder.add_edge(START, "determine_tool_and_initial_action")

# Sequential flow to avoid parallel state conflicts
builder.add_edge("determine_tool_and_initial_action", "research_step")
builder.add_edge("research_step", "reflection")

# Conditional routing after reflection
builder.add_conditional_edges(
    "reflection",
    evaluate_research,
    {
        "prepare_followup_research": "prepare_followup_research",
        "finalize_answer": "finalize_answer"
    }
)

# After preparing follow-up research, conditionally route based on completion
builder.add_conditional_edges(
    "prepare_followup_research",
    check_research_completion,
    {
        "research_step": "research_step",
        "finalize_answer": "finalize_answer"
    }
)

# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")