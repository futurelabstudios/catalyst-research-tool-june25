from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class FileSelection(BaseModel):
    selected_file_ids: List[str] = Field(
        description="A list of file IDs that are most relevant to the user's query."
    )
    rationale: str = Field(
        description="A brief explanation of why these files were chosen."
    )

class KBReflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided content from knowledge base files is sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    suggested_file_ids: List[str] = Field(
        description="A list of file IDs that might address the identified knowledge gaps."
    )