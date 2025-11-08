"""
FastAPI application for RAG backend.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

from .rag.graph import rag_graph
from .rag.vectorstore import VectorStoreManager
from .models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    HealthResponse,
    ErrorResponse,
    SourceDocument
)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation system built with LangGraph and FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store manager (singleton)
vector_manager = VectorStoreManager()


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("\n" + "="*80)
    print("RAG BACKEND API - Starting up...")
    print("="*80 + "\n")
    
    # Optionally load sample data if vector store is empty
    # Uncomment if you want automatic sample data loading
    # initialize_with_sample_data()
    
    print("✅ API ready to serve requests\n")
    print(f"📚 Docs available at: http://localhost:8000/docs")
    print(f"🔧 ReDoc available at: http://localhost:8000/redoc\n")


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check if the API and vector store are operational"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    try:
        # Check vector store
        vectorstore = vector_manager.vectorstore
        total_docs = len(vectorstore.docstore._dict)
        
        return HealthResponse(
            status="healthy",
            vectorstore_status="ready",
            total_documents=total_docs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


# Query endpoint
@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Query the RAG system",
    description="Ask a question and get an answer based on retrieved documents"
)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest containing the question and optional top_k
        
    Returns:
        QueryResponse with answer and source documents
    """
    try:
        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Run RAG pipeline
        result = rag_graph.invoke([HumanMessage(content=request.question)])
        
        # Extract answer and sources
        answer = result['messages'][-1].content
        sources = [
            SourceDocument(
                content=doc['content'],
                metadata=doc['metadata']
            )
            for doc in result['source_documents']
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.question
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# Document upload endpoint
@app.post(
    "/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload documents",
    description="Add new documents to the vector store"
)
async def upload_documents(request: DocumentUploadRequest):
    """
    Upload new documents to the vector store.
    
    Args:
        request: DocumentUploadRequest containing documents and optional metadata
        
    Returns:
        DocumentUploadResponse with upload statistics
    """
    try:
        # Validate documents
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided"
            )
        
        # Upload documents
        result = vector_manager.upload_documents(
            documents=request.documents,
            metadata=request.metadata if request.metadata else None
        )
        
        return DocumentUploadResponse(
            status=result['status'],
            documents_added=result['documents_added'],
            total_documents=result['total_documents']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading documents: {str(e)}"
        )


# Search endpoint (retrieval only, no generation)
@app.post(
    "/search",
    tags=["RAG"],
    summary="Search documents",
    description="Search for relevant documents without generating an answer"
)
async def search_documents(request: QueryRequest):
    """
    Search for relevant documents without generating an answer.
    
    Args:
        request: QueryRequest containing the search query and optional top_k
        
    Returns:
        Dictionary with search results
    """
    try:
        # Validate query
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        # Search documents
        docs = vector_manager.search(request.question, top_k=request.top_k if request.top_k else None)
        
        # Format results
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {
            "query": request.question,
            "documents": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}"
        )


# Clear vector store endpoint (use with caution!)
@app.delete(
    "/vectorstore",
    tags=["Documents"],
    summary="Clear vector store",
    description="⚠️ WARNING: Deletes all documents from the vector store"
)
async def clear_vectorstore():
    """
    Clear all documents from the vector store.
    
    ⚠️ WARNING: This action cannot be undone!
    
    Returns:
        Dictionary with status message
    """
    try:
        result = vector_manager.clear_vectorstore()
        
        return {
            "status": result['status'],
            "message": result['message'],
            "warning": "All documents have been deleted"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing vector store: {str(e)}"
        )


# Root endpoint
@app.get(
    "/",
    tags=["System"],
    summary="API Information",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "upload": "/upload",
            "search": "/search",
            "clear": "/vectorstore"
        }
    }


# Helper function to initialize with sample data
def initialize_with_sample_data():
    """Initialize vector store with sample documents."""
    sample_documents = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "LangGraph uses a graph structure where nodes represent actions and edges represent transitions.",
        "StateGraph is the main class in LangGraph for creating agent workflows.",
        "LangGraph supports checkpointing to save and restore agent state across sessions.",
        "Conditional edges in LangGraph allow dynamic routing based on agent decisions.",
        "LangGraph agents can use tools to interact with external systems and APIs.",
        "The MessagesState in LangGraph maintains conversation history using the add_messages reducer.",
        "LangGraph can visualize graphs using draw_ascii() and draw_mermaid() methods.",
        "The Command object in LangGraph allows nodes to control graph flow dynamically.",
        "LangGraph is built on top of LangChain and integrates seamlessly with it."
    ]
    
    metadata = [{"source": f"doc_{i}", "type": "sample"} for i in range(len(sample_documents))]
    
    result = vector_manager.upload_documents(sample_documents, metadata)
    print(f"✅ Initialized with {result['documents_added']} sample documents\n")


if __name__ == "__main__":
    import uvicorn
    
    # Optionally initialize with sample data
    # initialize_with_sample_data()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)