import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
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
    research_instructions, # Changed from web_searcher_instructions
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.internal_kb_tool import InternalKnowledgeBaseTool # NEW: Import internal KB tool

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Initialize the Internal Knowledge Base Tool
internal_kb_tool_instance = InternalKnowledgeBaseTool() # NEW: Instantiate the internal KB tool

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# Wrap the Google Search logic into a LangChain tool for consistency and easier use
@tool
def google_search_tool(query: str) -> dict:
    """Performs a Google search and returns a summarized result with citations.
    Returns a dictionary with 'content' and 'sources'."""
    # Note: In a production LangGraph setup, passing the full config to a tool
    # might require more robust patterns (e.g., using tool config or a wrapper).
    # For this example, we assume `query_generator_model` is suitable for generating search summaries.
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

    response = genai_client.models.generate_content(
        model=llm_for_search.model_name,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
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
    
    return {"content": modified_text, "sources": sources_gathered}


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

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
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the research node."""
    return [
        Send("research_step", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def research_step(state: WebSearchState, config: RunnableConfig) -> OverallState: # Renamed from web_research
    """LangGraph node that performs research using either internal KB or web search based on `use_web_search` flag."""
    configurable = Configuration.from_runnable_config(config)

    result_content = ""
    result_sources = []

    if configurable.use_web_search:
        print(f"DEBUG: Using Web Search for query: '{state['search_query']}'")
        tool_output = google_search_tool.invoke({"query": state["search_query"]})
        result_content = tool_output["content"]
        result_sources = tool_output["sources"]
    else:
        print(f"DEBUG: Using Internal Knowledge Base for query: '{state['search_query']}'")
        result_content = internal_kb_tool_instance.invoke(state["search_query"])
        result_sources = [] # Internal KB tool doesn't provide structured web citations

    return {
        "sources_gathered": result_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [result_content], # Keeping name for consistency in downstream nodes
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = configurable.reflection_model # Use model from config

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
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


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
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
        return [
            Send(
                "research_step", # Changed from web_research
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
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
    if configurable.use_web_search: # Only process web citations if web search was enabled
        for source in state["sources_gathered"]:
            # Check if source['short_url'] exists before replacement for robustness
            if "short_url" in source and source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)
    else:
        # For internal KB, sources_gathered is empty, no special handling needed here.
        # If internal KB should provide citations, modify internal_kb_tool to return them.
        pass

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("research_step", research_step) # Renamed node
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with research queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_research, ["research_step"]
)
# Reflect on the research
builder.add_edge("research_step", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["research_step", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")