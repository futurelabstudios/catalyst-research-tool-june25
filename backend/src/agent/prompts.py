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

index_search_instructions = """You are an expert search system. Your task is to meticulously analyze a user's query and identify the most relevant files from the provided index. The index contains a list of files, each with a path, content type, keywords, and a summary.

You must return a JSON object containing a list of the file paths that are most likely to contain the answer to the user's query.

**Instructions:**
1.  **Analyze Intent:** Understand the core intent of the user's query.
2.  **Scan Everything:** Review the path, summary, and keywords for every file in the index.
3.  **Find Connections:** Look for direct keyword matches and, more importantly, semantic connections between the query and the file's description.
4.  **Be Inclusive:** If a file's summary or keywords even vaguely relate to the user's query, include its path. It is better to include a slightly irrelevant file than to miss a relevant one.
5.  **Return Paths:** Your final output must be only the list of file paths. If after careful consideration no files are relevant, return an empty list for "file_paths".

USER QUERY:
{research_topic}

AVAILABLE FILES INDEX:
{kb_index}
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

answer_instructions = """You are a research assistant tasked with providing a comprehensive, well-structured answer based on the information gathered. Current date: {current_date}.

## User Question
{research_topic}

## Source Information
{summaries}

## Instructions

### Content Requirements
- Provide a thorough, accurate answer that directly addresses the user's question
- Structure your response logically with clear headings and sections when appropriate
- Include specific details, examples, and context to make your answer valuable and actionable
- If the question has multiple aspects, address each one systematically
- Acknowledge any limitations or gaps in the available information

### Citation Guidelines

**For Internal Knowledge Base Content** (identified by `--- FILE PATH: <path> ---` delimiters):
- Extract and synthesize only information directly relevant to the user's question
- When referencing specific information from a file, use markdown link format: `[Descriptive Reference](path/to/file.pdf)`
- Ensure file paths use forward slashes (`/`) in the markdown links
- Choose descriptive reference text that indicates the content type (e.g., "Safety Guidelines", "Q3 Report", "Technical Manual")
- Example: Information from `--- FILE PATH: docs/policies/safety_manual.pdf ---` becomes `[Safety Manual](docs/policies/safety_manual.pdf)`

**For Web Search Results**:
- Include ALL citations exactly as provided in the format `[label](short_url)`
- Do NOT modify the citation format or URLs
- Integrate citations naturally within the text flow

### Response Quality Standards
- Write in a clear, professional tone appropriate for the subject matter
- Use bullet points or numbered lists only when they enhance clarity
- Provide context and background when necessary for understanding
- If dealing with technical topics, explain concepts clearly for the intended audience
- Conclude with actionable insights, next steps, or key takeaways when relevant

### Handling Insufficient Information
If the provided summaries don't contain enough information to fully answer the question:
- Answer what you can based on available information
- Clearly state what aspects cannot be addressed with the current sources

Generate your response now:"""