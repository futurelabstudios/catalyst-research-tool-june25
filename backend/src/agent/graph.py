import os
from typing import List, AsyncGenerator
import asyncio


from agent.tools_and_schemas import SearchQueryList, Reflection, ChosenKBTopic
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
    kb_router_instructions,
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

# Initialize the Internal Knowledge Base Tool (now scans directory)
internal_kb_tool_instance = InternalKnowledgeBaseTool()

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
        return {"query_list": result.query, "use_web_search": True}
    else:
        print("DEBUG: Using Internal Knowledge Base path. Routing to KB topic selection.")
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model, # Using query_generator_model for routing as well
            temperature=0.0, # Keep temperature low for routing
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        structured_llm = llm.with_structured_output(ChosenKBTopic)

        available_topics_str = internal_kb_tool_instance.get_available_topics_for_llm()
        formatted_prompt = kb_router_instructions.format(
            available_kb_topics=available_topics_str,
            research_topic=user_query,
        )
        
        try:
            chosen_topic_result = structured_llm.invoke(formatted_prompt)
            chosen_topic = chosen_topic_result.topic
            print(f"DEBUG: LLM chose internal KB topic: '{chosen_topic}'")
            # We will use 'search_query' field to store the chosen KB topic for research_step to consume
            # This allows research_step to keep a consistent input state type.
            return {"search_query": [chosen_topic], "use_web_search": False, "chosen_kb_topic": chosen_topic}
        except Exception as e:
            print(f"ERROR: Failed to route internal KB query: {e}. Returning an error message.")
            # Fallback if LLM fails to choose a topic
            return {
                "web_research_result": [f"I could not determine a relevant internal knowledge base topic for your query. Error: {e}. Available topics: {available_topics_str}"],
                "messages": [AIMessage(content=f"I could not determine a relevant internal knowledge base topic for your query. Error: {e}. Available topics: {available_topics_str}")],
                "use_web_search": False, # Still indicate internal KB was attempted
                "chosen_kb_topic": "error_fallback"
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
                "research_loop_count": state.get("research_loop_count", 0) + 1,
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
    
    else: # Internal KB path (already sequential, but we can stream progress)
        for topic in current_loop_queries:
            yield {"messages": [AIMessage(content=f"Retrieving from Internal KB: '{topic}'", name="tool_code", tool_call_id="research_update")]}
            await asyncio.sleep(0.5) # Simulate work and allow message to show
            kb_content = await internal_kb_tool_instance.ainvoke(topic, config=config)
            gathered_contents.append(kb_content)
            ran_queries.append(topic)
            yield {"messages": [AIMessage(content=f"Finished retrieving from Internal KB: '{topic}'", name="tool_code", tool_call_id="research_update")]}

    # Yield the final accumulated state for this node
    yield {
        "sources_gathered": gathered_sources,
        "search_query": ran_queries,
        "web_research_result": gathered_contents,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
        "number_of_ran_queries": len(ran_queries),
    }



def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
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
) -> str | List[Send]: # Return type changed for clarity
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
        # If not sufficient and not max loops, continue researching
        if configurable.use_web_search:
            # For web search, generate new queries
            return [
                Send(
                    "research_step",
                    {
                        "query_list": state["follow_up_queries"], # Pass follow-up queries to research_step
                        "id": state["number_of_ran_queries"] + int(idx),
                    },
                )
                for idx, follow_up_query in enumerate(state["follow_up_queries"])
            ]
        else:
            # For internal KB, we don't generate follow-up search queries.
            # The reflection suggested a knowledge gap, but we already sent the whole relevant file.
            # We should probably go directly to finalize_answer here for internal KB,
            # as the current setup doesn't support iterative KB refinement from follow-up queries.
            # Or, we could attempt to re-route to a *different* KB topic if the reflection implies
            # a different area of the KB might be relevant. For now, let's finalize.
            print("DEBUG: Internal KB: Reflection indicated knowledge gap, but current flow doesn't support iterative KB search. Finalizing answer.")
            return "finalize_answer"


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


# Update the OverallState definition in state.py
# If you haven't done this already, update backend/src/agent/state.py
# by adding 'chosen_kb_topic: Optional[str]' to OverallState.
# (If you followed the previous step to add this, you can skip this comment block,
# but it's crucial for the state to correctly propagate the chosen topic).

# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
# NEW: Initial node to determine the path (web search vs. internal KB)
builder.add_node("determine_tool_and_initial_action", determine_tool_and_initial_action)
builder.add_node("research_step", research_step)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `determine_tool_and_initial_action`
builder.add_edge(START, "determine_tool_and_initial_action")

# Conditional routing AFTER determine_tool_and_initial_action
# If use_web_search is True, proceed to research_step immediately.
# If use_web_search is False, research_step will internally handle KB topic.
builder.add_edge("determine_tool_and_initial_action", "research_step")


# Reflect on the research (same for both paths)
builder.add_edge("research_step", "reflection")

# Evaluate the research and decide next step
builder.add_conditional_edges(
    "reflection",
    evaluate_research,
    {
        "research_step": "research_step", # For web search, go back to research_step with follow-up queries
        "finalize_answer": "finalize_answer" # For KB or if sufficient, finalize
    }
)

# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")