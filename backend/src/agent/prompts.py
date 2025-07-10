from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format:
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}Context: {research_topic}"""

# NEW PROMPT for Internal KB Routing
kb_router_instructions = """You are an intelligent router responsible for directing user queries to the most relevant internal knowledge base topic.
You will be given a user's query and a list of available knowledge base topics.
Your task is to select *one single* topic that is most likely to contain the information needed to answer the user's query.

Instructions:
- Analyze the user's query carefully.
- Choose the single most relevant topic from the `available_kb_topics` list.
- If no topic seems directly relevant, choose the topic that is broadly closest, or a general topic if one exists (e.g., 'general_info').
- Only output the topic name as it appears in the `available_kb_topics` list. Do NOT invent new topics.

Format:
- Format your response as a JSON object with these exact keys:
   - "topic": The chosen knowledge base topic (e.g., "gender", "education").
   - "rationale": A brief explanation of why this topic was chosen.

Available knowledge base topics: {available_kb_topics}
User Query: {research_topic}
"""

research_instructions = """You are an expert research assistant. Conduct targeted research based on the provided topic and summarize the findings into a verifiable text artifact.
Instructions:
The current date is {current_date}.
Consolidate key findings while meticulously tracking the source(s) for each specific piece of information (if applicable, e.g., for web search, internal file path).
IF THE PROVIDED SUMMARIES ARE FROM THE INTERNAL KNOWLEDGE BASE: You have been provided with the *entire* content of a specific internal knowledge base file, formatted with '--- FILE PATH: <path> ---' delimiters. Your task is to identify and extract *only* the information highly relevant to the "Research Topic" from this full content.
The output should be a well-written summary or report based on your research findings.
Only include the information found in the research results, don't make up any information.
Research Topic:
{research_topic}
"""
reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".
Instructions:
Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.
Requirements:
Ensure the follow-up query is self-contained and includes necessary context for research.
Output Format:
Format your response as a JSON object with these exact keys:
"is_sufficient": true or false
"knowledge_gap": Describe what information is missing or needs clarification
"follow_up_queries": Write a specific question to address this gap
Example:
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
Use code with caution.
Json
Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
Summaries:
{summaries}"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.
Instructions:
The current date is {current_date}.
You are the final step of a multi-step research process, don't mention that you are the final step.
You have access to all the information gathered from the previous steps.
You have access to the user's question.
Generate a high-quality answer to the user's question based on the provided summaries and the user's question.

IF THE SUMMARIES ARE FROM THE INTERNAL KNOWLEDGE BASE: The provided summaries contain the entire content of a specific internal knowledge base file, formatted with '--- FILE PATH: <path> ---' delimiters.
- Extract and synthesize only the information directly relevant to the user's question from this content.
- When referencing information that clearly comes from a specific file, use markdown link format: `[Descriptive reference/filename](path/to/file.pdf)`.
- Ensure the `path` in the markdown link uses `/` (forward slashes).
- Example: If the source is `--- FILE PATH: docs/reports/my_report.pdf ---`, you could cite it as `[My Report](docs/reports/my_report.pdf)`.
- You do not need to include traditional numerical web citations for internal KB content.

FOR WEB SEARCH RESULTS:
- you MUST include all the citations from the summaries in the answer correctly. These will already be provided to you in the format `[label](short_url)`. Do not alter this format.

User Context:
{research_topic}
Summaries:
{summaries}"""


kb_file_selector_instructions = """You are an intelligent file selector for an internal knowledge base. 
You will be given a user's query and an index of available files with summaries.
Your task is to select the most relevant file_id(s) that are likely to contain information needed to answer the user's query.

Instructions:
- Analyze the user's query and the file summaries carefully
- Select 1-3 file_ids that are most relevant to the query
- Focus on files that directly address the user's question
- Consider keywords, content type, and file descriptions

Format your response as a JSON object with these exact keys:
- "selected_file_ids": List of chosen file_ids (e.g., ["file123", "file456"])
- "rationale": Brief explanation of why these files were chosen

Available Files Index:
{files_index}

User Query: {user_query}
"""

kb_reflection_instructions = """You are an expert research assistant analyzing content from internal knowledge base files about "{user_query}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration
- If the provided content is sufficient to answer the user's question, indicate so
- If there are knowledge gaps, suggest specific file_id(s) from the available index that might fill those gaps
- Focus on missing technical details, implementation specifics, or related information

Requirements:
- Only suggest file_ids that exist in the provided index
- Ensure suggested files are likely to contain complementary information

Format your response as a JSON object with these exact keys:
- "is_sufficient": true or false
- "knowledge_gap": Describe what information is missing (empty string if sufficient)
- "suggested_file_ids": List of file_ids that might address the gaps (empty list if sufficient)

Available Files Index:
{files_index}

Current Content Analysis:
{summaries}
"""
