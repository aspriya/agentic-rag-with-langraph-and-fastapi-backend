from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is StateGraph in LangGraph?",
                "top_k": 3
            }
        }


class SourceDocument(BaseModel):
    """Model for a source document."""
    content: str = Field(..., description="Document content preview")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Retrieved source documents")
    query: str = Field(..., description="Original query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "StateGraph is the main class in LangGraph for creating agent workflows...",
                "sources": [
                    {
                        "content": "StateGraph is the main class in LangGraph...",
                        "metadata": {"source": "doc_1", "type": "sample"}
                    }
                ],
                "query": "What is StateGraph in LangGraph?"
            }
        }


class DocumentUploadRequest(BaseModel):
    """Request model for uploading documents."""
    documents: List[str] = Field(..., min_length=1, description="List of document texts")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata for each document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    "LangGraph is a library for building stateful applications.",
                    "Multi-agent systems can coordinate multiple AI agents."
                ],
                "metadata": [
                    {"source": "manual", "category": "overview"},
                    {"source": "manual", "category": "architecture"}
                ]
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    status: str = Field(..., description="Upload status")
    documents_added: int = Field(..., description="Number of documents added")
    total_documents: int = Field(..., description="Total documents in vector store")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "documents_added": 5,
                "total_documents": 25
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    vectorstore_status: str = Field(..., description="Vector store status")
    total_documents: int = Field(..., description="Total documents in store")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "vectorstore_status": "ready",
                "total_documents": 25
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Question cannot be empty"
            }
        }
