from typing import TypedDict, List, Optional
from langgraph.graph import MessagesState


class RAGState(MessagesState):
    """
    State for RAG pipeline.
    
    Attributes:
        messages: Conversation history (inherited from MessagesState)
        context: Retrieved document text
        source_documents: Metadata about retrieved documents
        query: Original user query
    """
    context: str
    source_documents: Optional[List[dict]] = None
    query: Optional[str] = None