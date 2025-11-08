# Creating a RAG Backend using LangGraph & FastAPI

A step-by-step guide to building a production-ready RAG (Retrieval-Augmented Generation) system with document upload capabilities.

---

## Table of Contents

1. [Project Structure Setup](#project-structure-setup)
2. [RAG Implementation](#rag-implementation)
3. [FastAPI Integration](#fastapi-integration) *(Coming next)*

---

## Part 1: Project Structure Setup

### Step 1: Create Project Directory Structure

Create the following folder structure:

```
rag-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app (later)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph RAG pipeline
│   │   ├── nodes.py         # Node functions
│   │   ├── state.py         # State definitions
│   │   └── vectorstore.py   # Vector store management
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models (later)
├── data/
│   ├── documents/           # Original documents
│   └── vectorstore/         # FAISS index storage
├── .env                     # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

**Create the project:**

```bash
# Create main directory
mkdir rag-backend
cd rag-backend

# Create subdirectories
mkdir -p app/rag app/models data/documents data/vectorstore

# Create __init__.py files
New-Item -ItemType File -Path app/__init__.py
New-Item -ItemType File -Path app/rag/__init__.py
New-Item -ItemType File -Path app/models/__init__.py
```

---

### Step 2: Setup Environment File

Create `.env` file in the project root:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Configuration
VECTORSTORE_PATH=./data/vectorstore
DOCUMENTS_PATH=./data/documents

# LLM Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0

# Retrieval Configuration
TOP_K_DOCUMENTS=3
```

---

### Step 3: Create Requirements File

Create `requirements.txt`:

```txt
# Core Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
python-dotenv>=1.0.0

# LangChain & LangGraph
langchain>=0.1.0
langchain-openai>=0.0.2
langchain-community>=0.0.10
langgraph>=0.0.20

# Vector Store
faiss-cpu>=1.9.0

# Utilities
pydantic>=2.5.0
python-multipart>=0.0.6
```

**Install dependencies:**

```bash
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

---

### Step 4: Create .gitignore

Create `.gitignore`:

```gitignore
# Environment
.env
venv/
__pycache__/
*.pyc

# Vector Store Data
data/vectorstore/*
!data/vectorstore/.gitkeep

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Part 2: RAG Implementation

Now let's implement the RAG system step by step.

---

### Step 1: Define State (`app/rag/state.py`)

**Why**: State holds conversation messages, retrieved context, and metadata throughout the RAG pipeline.

**What**: Create a custom state that extends MessagesState with context and document metadata.

```python
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
```

**Key Points:**
- `messages`: Automatically handles conversation history with `add_messages` reducer
- `context`: Stores concatenated text from retrieved documents
- `source_documents`: Tracks which documents were used (for citations)
- `query`: Preserves original user question for logging/debugging

---

### Step 2: Vector Store Manager (`app/rag/vectorstore.py`)

**Why**: Centralized management of document embeddings, storage, and retrieval with upload capability.

**What**: Create a class to handle FAISS vector store operations including document uploads.

```python
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    """Manages FAISS vector store for document storage and retrieval."""
    
    def __init__(
        self,
        vectorstore_path: str = None,
        embedding_model: str = None
    ):
        self.vectorstore_path = vectorstore_path or os.getenv(
            "VECTORSTORE_PATH", 
            "./data/vectorstore"
        )
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small"
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # Load or create vector store
        self.vectorstore = self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self) -> FAISS:
        """Load existing vector store or create empty one."""
        index_path = os.path.join(self.vectorstore_path, "index")
        
        if os.path.exists(index_path + ".faiss"):
            print(f"✓ Loading existing vector store from {self.vectorstore_path}")
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"✓ Creating new vector store at {self.vectorstore_path}")
            # Create with dummy document (FAISS requires at least one)
            dummy_doc = Document(page_content="Initialization document")
            vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            
            # Create directory if needed
            os.makedirs(self.vectorstore_path, exist_ok=True)
            vectorstore.save_local(self.vectorstore_path)
            
            return vectorstore
    
    def upload_documents(self, documents: List[str], metadata: List[dict] = None) -> dict:
        """
        Upload new documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            dict with upload statistics
        """
        if not documents:
            return {"status": "error", "message": "No documents provided"}
        
        # Create Document objects
        docs = []
        for i, text in enumerate(documents):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            docs.append(Document(page_content=text, metadata=meta))
        
        # Add to vector store
        self.vectorstore.add_documents(docs)
        
        # Save updated vector store
        self.vectorstore.save_local(self.vectorstore_path)
        
        print(f"✓ Uploaded {len(docs)} documents to vector store")
        
        return {
            "status": "success",
            "documents_added": len(docs),
            "total_documents": len(self.vectorstore.docstore._dict)
        }
    
    def get_retriever(self, top_k: int = None):
        """
        Get a retriever for the vector store.
        
        Args:
            top_k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        k = top_k or int(os.getenv("TOP_K_DOCUMENTS", "3"))
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def search(self, query: str, top_k: int = None) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        k = top_k or int(os.getenv("TOP_K_DOCUMENTS", "3"))
        return self.vectorstore.similarity_search(query, k=k)
    
    def clear_vectorstore(self) -> dict:
        """Clear all documents from vector store (use with caution!)."""
        # Create fresh vector store
        dummy_doc = Document(page_content="Initialization document")
        self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        self.vectorstore.save_local(self.vectorstore_path)
        
        print("⚠️  Vector store cleared")
        return {"status": "success", "message": "Vector store cleared"}
```

**Key Features:**
- **Persistent Storage**: Saves FAISS index to disk, loads on restart
- **Upload Documents**: Add new documents anytime without rebuilding
- **Flexible Retrieval**: Configurable top-k results
- **Metadata Support**: Track document sources, timestamps, etc.
- **Error Handling**: Graceful initialization and updates

---

### Step 3: Node Functions (`app/rag/nodes.py`)

**Why**: Nodes perform the actual retrieval and generation work.

**What**: Create retrieval and generation nodes that use the vector store manager.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .state import RAGState
from .vectorstore import VectorStoreManager
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize vector store manager (singleton pattern)
vector_manager = VectorStoreManager()


def retrieve_documents(state: RAGState) -> dict:
    """
    Retrieval Node: Searches vector store for relevant documents.
    
    Process:
        1. Extract user's question from messages
        2. Query vector store for top-k relevant docs
        3. Combine document content into context string
        4. Store metadata for citation purposes
    """
    # Get the last user message
    last_message = state["messages"][-1].content
    
    # Search vector store
    docs = vector_manager.search(last_message)
    
    # Combine retrieved docs into context string
    context = "\n\n".join([
        f"[Document {i+1}]\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    # Extract metadata for citations
    source_documents = [
        {
            "content": doc.page_content[:200] + "...",  # Preview
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
    print(f"🔍 Retrieved {len(docs)} documents for query: '{last_message[:50]}...'")
    
    return {
        "context": context,
        "source_documents": source_documents,
        "query": last_message
    }


def generate_answer(state: RAGState) -> dict:
    """
    Generation Node: Creates answer using retrieved context.
    
    Process:
        1. Build RAG prompt with context
        2. Call LLM with context + user question
        3. Return generated answer
    """
    # Create RAG prompt
    system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the following context.

IMPORTANT INSTRUCTIONS:
- Base your answer ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- If relevant, mention which document(s) support your answer

CONTEXT:
{state['context']}
"""
    
    # Get user's question
    user_question = state["query"] or state["messages"][-1].content
    
    # Call LLM
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ])
    
    print(f"💬 Generated answer (length: {len(response.content)} chars)")
    
    return {"messages": [response]}


def upload_documents_node(state: RAGState) -> dict:
    """
    Upload Node: Adds new documents to vector store.
    
    Note: This node expects documents in state (added by API endpoint)
    """
    # This is a placeholder - actual upload is handled by API
    # In a graph, you might route here based on user intent
    return {}
```

**Design Decisions:**
- **Singleton Vector Manager**: One instance shared across all nodes (efficient)
- **Document Numbering**: Context includes "[Document 1]" markers for citation
- **Metadata Preservation**: Source tracking for transparency
- **Configurable LLM**: Model and temperature from environment variables
- **Defensive Prompting**: Instructs LLM to admit when context is insufficient

---

### Step 4: Build RAG Graph (`app/rag/graph.py`)

**Why**: Graph orchestrates the retrieval → generation pipeline.

**What**: Create a LangGraph that connects retrieve and generate nodes.

```python
from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .nodes import retrieve_documents, generate_answer


class RAGGraph:
    """LangGraph-based RAG pipeline."""
    
    def __init__(self):
        self.app = self._build_graph()
    
    def _build_graph(self):
        """Build the RAG graph: START → retrieve → generate → END"""
        
        # Create graph
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("retrieve", retrieve_documents)
        graph.add_node("generate", generate_answer)
        
        # Add edges (linear flow)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        
        # Compile
        return graph.compile()
    
    def invoke(self, messages: list) -> dict:
        """
        Run the RAG pipeline.
        
        Args:
            messages: List of message objects (HumanMessage, AIMessage, etc.)
            
        Returns:
            Final state with answer and metadata
        """
        result = self.app.invoke({
            "messages": messages,
            "context": "",
            "source_documents": None,
            "query": None
        })
        return result
    
    def visualize(self) -> str:
        """Return ASCII visualization of the graph."""
        return self.app.get_graph().draw_ascii()


# Global instance (singleton)
rag_graph = RAGGraph()
```

**Architecture:**
- **Simple Pipeline**: Linear flow works for most RAG use cases
- **Stateful Execution**: State accumulates through retrieve → generate
- **Singleton Pattern**: One graph instance serves all requests (efficient)
- **Visualization Support**: Helpful for debugging and documentation

---

### Step 5: Create Main Entry Point (`app/main.py`)

**Why**: Provides a simple interface to test RAG functionality before adding FastAPI.

**What**: CLI tool to upload documents and query the RAG system.

```python
"""
Main entry point for RAG backend.
FastAPI integration coming in Part 3.
"""

from langchain_core.messages import HumanMessage
from rag.graph import rag_graph
from rag.vectorstore import VectorStoreManager
import os
from dotenv import load_dotenv

load_dotenv()


def initialize_with_sample_data():
    """Upload sample documents if vector store is empty."""
    
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
    
    vector_manager = VectorStoreManager()
    result = vector_manager.upload_documents(sample_documents, metadata)
    
    print(f"\n✅ Initialized with {result['documents_added']} sample documents")
    print(f"Total documents in store: {result['total_documents']}\n")


def test_rag_query(question: str):
    """Test the RAG pipeline with a question."""
    
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    # Run RAG pipeline
    result = rag_graph.invoke([HumanMessage(content=question)])
    
    # Display results
    print(f"ANSWER:\n{result['messages'][-1].content}\n")
    
    print(f"RETRIEVED DOCUMENTS:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n  [{i}] {doc['content']}")
        if doc['metadata']:
            print(f"      Metadata: {doc['metadata']}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main CLI interface."""
    
    print("\n" + "="*80)
    print("RAG BACKEND - LangGraph + FastAPI")
    print("="*80 + "\n")
    
    # Initialize with sample data
    initialize_with_sample_data()
    
    # Visualize graph
    print("RAG PIPELINE STRUCTURE:")
    print(rag_graph.visualize())
    print()
    
    # Test queries
    test_questions = [
        "What is StateGraph in LangGraph?",
        "Does LangGraph support checkpointing?",
        "How can I visualize a LangGraph workflow?"
    ]
    
    for question in test_questions:
        test_rag_query(question)


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Run the RAG system
python -m app.main
```

**What It Does:**
1. Loads sample documents into vector store
2. Visualizes the RAG graph structure
3. Runs test queries to demonstrate retrieval + generation
4. Shows retrieved documents and metadata

---

### Step 6: Test the RAG System

**Create a test script** (`test_rag.py` in project root):

```python
"""
Test script for RAG functionality.
"""

from langchain_core.messages import HumanMessage
from app.rag.graph import rag_graph
from app.rag.vectorstore import VectorStoreManager


def test_document_upload():
    """Test uploading custom documents."""
    
    print("\n=== TEST: Document Upload ===\n")
    
    vector_manager = VectorStoreManager()
    
    # Upload custom documents
    custom_docs = [
        "The LangGraph library was created by LangChain to address the need for stateful agent workflows.",
        "Multi-agent systems in LangGraph can have supervisor agents that coordinate worker agents.",
        "LangGraph supports both synchronous and asynchronous execution modes."
    ]
    
    metadata = [
        {"source": "custom_1", "category": "overview"},
        {"source": "custom_2", "category": "architecture"},
        {"source": "custom_3", "category": "features"}
    ]
    
    result = vector_manager.upload_documents(custom_docs, metadata)
    print(f"✅ Upload result: {result}\n")


def test_rag_pipeline():
    """Test end-to-end RAG query."""
    
    print("\n=== TEST: RAG Pipeline ===\n")
    
    question = "Tell me about supervisor agents in LangGraph"
    result = rag_graph.invoke([HumanMessage(content=question)])
    
    print(f"Question: {question}\n")
    print(f"Answer: {result['messages'][-1].content}\n")
    print(f"Sources: {len(result['source_documents'])} documents retrieved\n")


def test_retrieval_only():
    """Test retrieval without generation."""
    
    print("\n=== TEST: Retrieval Only ===\n")
    
    vector_manager = VectorStoreManager()
    
    query = "checkpointing"
    docs = vector_manager.search(query, top_k=3)
    
    print(f"Query: '{query}'\n")
    print(f"Retrieved {len(docs)} documents:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    test_document_upload()
    test_rag_pipeline()
    test_retrieval_only()
    print("\n✅ All tests completed!\n")
```

**Run tests:**
```bash
python test_rag.py
```

---

## Summary: What We've Built So Far

✅ **Complete RAG System with:**

1. ✅ **Project Structure**: Organized, scalable folder layout
2. ✅ **Environment Configuration**: .env-based settings
3. ✅ **State Management**: Custom RAGState with context tracking
4. ✅ **Vector Store**: Persistent FAISS storage with upload capability
5. ✅ **Retrieval Node**: Semantic search with metadata
6. ✅ **Generation Node**: Context-aware LLM responses
7. ✅ **LangGraph Pipeline**: Orchestrated retrieve → generate flow
8. ✅ **Document Upload**: Add new documents dynamically
9. ✅ **Testing Tools**: CLI and test scripts

### Key Features

- **Persistent Storage**: Vector store survives restarts
- **Dynamic Updates**: Upload documents without rebuilding
- **Metadata Tracking**: Source attribution for citations
- **Configurable**: Environment-based settings
- **Production-Ready Structure**: Modular, testable, maintainable

---

---

## Part 3: FastAPI Integration

Now let's add REST API endpoints to make our RAG system accessible via HTTP.

---

### Step 1: Define Pydantic Models (`app/models/schemas.py`)

**Why**: Type-safe request/response validation and automatic API documentation.

**What**: Create schemas for query requests, responses, and document uploads.

```python
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
    documents: List[str] = Field(..., min_items=1, description="List of document texts")
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
```

**Key Design Decisions:**
- **Field Validation**: Min/max lengths, ranges for top_k
- **Examples**: Swagger UI shows realistic request/response examples
- **Optional Fields**: Metadata is optional for flexibility
- **Type Safety**: Pydantic ensures data integrity

---

### Step 2: Create FastAPI Application (`app/main.py`)

**Why**: Expose RAG functionality via REST API endpoints.

**What**: Replace the CLI version with a full FastAPI application.

```python
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
        
        # Extract answer
        answer = result['messages'][-1].content
        
        # Format source documents
        sources = [
            SourceDocument(
                content=doc['content'],
                metadata=doc['metadata']
            )
            for doc in result.get('source_documents', [])
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
        
        # Check if metadata matches documents count
        if request.metadata and len(request.metadata) != len(request.documents):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Metadata count must match documents count"
            )
        
        # Upload documents
        result = vector_manager.upload_documents(
            documents=request.documents,
            metadata=request.metadata
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
    tags=["Documents"],
    summary="Search documents",
    description="Search for relevant documents without generating an answer"
)
async def search_documents(request: QueryRequest):
    """
    Search for relevant documents without generating an answer.
    
    Args:
        request: QueryRequest containing the search query
        
    Returns:
        List of relevant documents
    """
    try:
        # Search vector store
        docs = vector_manager.search(request.question, top_k=request.top_k)
        
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
        Status message
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
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "upload": "POST /upload",
            "search": "POST /search",
            "clear": "DELETE /vectorstore"
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
    
    metadata = [{"source": f"sample_{i}", "type": "initialization"} for i in range(len(sample_documents))]
    
    result = vector_manager.upload_documents(sample_documents, metadata)
    print(f"✅ Initialized with {result['documents_added']} sample documents\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )
```

**Key Features:**
- **Comprehensive Endpoints**: Query, upload, search, health check, clear
- **Error Handling**: Global exception handler + endpoint-specific validation
- **CORS Support**: Configurable cross-origin requests
- **Auto Documentation**: Swagger UI at `/docs`, ReDoc at `/redoc`
- **Startup Initialization**: Optional sample data loading
- **Type Safety**: Pydantic validation on all requests/responses

---

### Step 3: Create API Testing Script (`test_api.py`)

**Why**: Verify API endpoints work correctly.

**What**: Python script to test all endpoints programmatically.

```python
"""
API testing script for RAG backend.
Run the API first: uvicorn app.main:app --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_health():
    """Test health check endpoint."""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_upload_documents():
    """Test document upload endpoint."""
    print_section("TEST 2: Upload Documents")
    
    payload = {
        "documents": [
            "LangGraph supports both synchronous and asynchronous execution.",
            "The StateGraph class allows you to define complex agent workflows.",
            "LangGraph integrates seamlessly with LangChain tools and chains."
        ],
        "metadata": [
            {"source": "test_doc_1", "category": "features"},
            {"source": "test_doc_2", "category": "architecture"},
            {"source": "test_doc_3", "category": "integration"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/upload", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_search_documents():
    """Test search endpoint."""
    print_section("TEST 3: Search Documents")
    
    payload = {
        "question": "What is StateGraph?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_query_rag():
    """Test RAG query endpoint."""
    print_section("TEST 4: RAG Query")
    
    payload = {
        "question": "How does LangGraph integrate with LangChain?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    
    result = response.json()
    print(f"\nQuestion: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])} documents):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  [{i}] {source['content'][:100]}...")
        print(f"      Metadata: {source['metadata']}")


def test_invalid_request():
    """Test error handling with invalid request."""
    print_section("TEST 5: Error Handling")
    
    # Empty question
    payload = {"question": ""}
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_multiple_queries():
    """Test multiple queries in sequence."""
    print_section("TEST 6: Multiple Queries")
    
    questions = [
        "What is LangGraph?",
        "Does LangGraph support checkpointing?",
        "How can I visualize workflows in LangGraph?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Query {i} ---")
        payload = {"question": question, "top_k": 2}
        
        response = requests.post(f"{BASE_URL}/query", json=payload)
        result = response.json()
        
        print(f"Q: {question}")
        print(f"A: {result['answer'][:150]}...")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  RAG BACKEND API - TESTING SUITE")
    print("="*80)
    print("\nMake sure the API is running: uvicorn app.main:app --reload\n")
    
    try:
        # Run tests
        test_health()
        test_upload_documents()
        test_search_documents()
        test_query_rag()
        test_invalid_request()
        test_multiple_queries()
        
        print("\n" + "="*80)
        print("  ✅ ALL TESTS COMPLETED")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API. Make sure it's running!")
        print("   Run: uvicorn app.main:app --reload\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Terminal 1: Start API
uvicorn app.main:app --reload

# Terminal 2: Run tests
python test_api.py
```

---

### Step 4: Create Curl Examples (`api_examples.md`)

**Why**: Quick reference for testing with curl or other HTTP clients.

**What**: Create a markdown file with example API calls.

```markdown
# API Examples

Quick reference for testing the RAG Backend API.

## Base URL
```
http://localhost:8000
```

---

## 1. Health Check

**Check if API is running:**

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "vectorstore_status": "ready",
  "total_documents": 15
}
```

---

## 2. Upload Documents

**Upload new documents to vector store:**

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "LangGraph is a library for building stateful applications with LLMs.",
      "Multi-agent systems in LangGraph enable complex workflows."
    ],
    "metadata": [
      {"source": "manual", "category": "overview"},
      {"source": "manual", "category": "features"}
    ]
  }'
```

**Response:**
```json
{
  "status": "success",
  "documents_added": 2,
  "total_documents": 17
}
```

---

## 3. Search Documents (Retrieval Only)

**Search without generating an answer:**

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LangGraph?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "query": "What is LangGraph?",
  "documents": [
    {
      "content": "LangGraph is a library for building stateful applications...",
      "metadata": {"source": "doc_1"}
    }
  ],
  "count": 3
}
```

---

## 4. Query RAG System

**Ask a question and get AI-generated answer:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does LangGraph handle state?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "answer": "LangGraph handles state through the StateGraph class...",
  "sources": [
    {
      "content": "StateGraph is the main class in LangGraph...",
      "metadata": {"source": "doc_1", "type": "sample"}
    }
  ],
  "query": "How does LangGraph handle state?"
}
```

---

## 5. Clear Vector Store (Dangerous!)

**⚠️ Delete all documents:**

```bash
curl -X DELETE "http://localhost:8000/vectorstore"
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector store cleared",
  "warning": "All documents have been deleted"
}
```

---

## PowerShell Examples

**Health Check:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**Upload Documents:**
```powershell
$body = @{
    documents = @(
        "LangGraph is a library for building stateful applications.",
        "Multi-agent systems enable complex workflows."
    )
    metadata = @(
        @{source = "manual"; category = "overview"}
        @{source = "manual"; category = "features"}
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/upload" -Method Post -Body $body -ContentType "application/json"
```

**Query RAG:**
```powershell
$query = @{
    question = "What is LangGraph?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -Body $query -ContentType "application/json"
```

---

## Python Requests Examples

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Upload documents
payload = {
    "documents": ["Document 1", "Document 2"],
    "metadata": [{"source": "test"}, {"source": "test"}]
}
response = requests.post("http://localhost:8000/upload", json=payload)
print(response.json())

# Query RAG
payload = {"question": "What is LangGraph?", "top_k": 3}
response = requests.post("http://localhost:8000/query", json=payload)
result = response.json()
print(f"Answer: {result['answer']}")
```

---

## Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide interactive testing interfaces!
```

Save this as `api_examples.md` in the project root.

---

### Step 5: Run the FastAPI Application

**Development Mode (with auto-reload):**

```bash
# From project root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using Python directly:**

```bash
python -m app.main
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.

================================================================================
RAG BACKEND API - Starting up...
================================================================================

✅ API ready to serve requests

📚 Docs available at: http://localhost:8000/docs
🔧 ReDoc available at: http://localhost:8000/redoc

INFO:     Application startup complete.
```

---

### Step 6: Test via Swagger UI

**Access Interactive Documentation:**

1. Open browser: `http://localhost:8000/docs`
2. You'll see all endpoints with "Try it out" buttons
3. Click "Try it out" on `/query` endpoint
4. Enter a question in the request body:
   ```json
   {
     "question": "What is StateGraph in LangGraph?",
     "top_k": 3
   }
   ```
5. Click "Execute"
6. View the response with answer and sources

**Test Document Upload:**

1. Click `/upload` endpoint
2. Click "Try it out"
3. Enter documents:
   ```json
   {
     "documents": [
       "LangGraph supports multi-agent coordination.",
       "Supervisor patterns enable hierarchical agent systems."
     ],
     "metadata": [
       {"source": "manual", "category": "features"},
       {"source": "manual", "category": "patterns"}
     ]
   }
   ```
4. Click "Execute"
5. Verify successful upload

---

### Step 7: Create Docker Configuration (Optional)

**Why**: Containerize the application for easy deployment.

**What**: Create Dockerfile and docker-compose.yml.

**Dockerfile:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env .env

# Create data directories
RUN mkdir -p data/documents data/vectorstore

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-backend:
    build: .
    container_name: rag-backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTORSTORE_PATH=/app/data/vectorstore
      - DOCUMENTS_PATH=/app/data/documents
      - LLM_MODEL=gpt-4o-mini
      - EMBEDDING_MODEL=text-embedding-3-small
    volumes:
      - ./data:/app/data
      - ./app:/app/app
    restart: unless-stopped
```

**Build and Run:**

```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

---

### Step 8: Add Environment Variables for Production

**Update `.env` for production settings:**

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Configuration
VECTORSTORE_PATH=./data/vectorstore
DOCUMENTS_PATH=./data/documents

# LLM Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0

# Retrieval Configuration
TOP_K_DOCUMENTS=3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS Settings (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Security (optional)
API_KEY=your_secret_api_key_here  # For authentication if needed
```

---

## Summary: Complete FastAPI Integration

✅ **What We Added:**

1. ✅ **Pydantic Schemas** - Type-safe request/response models
2. ✅ **FastAPI Application** - Complete REST API with 6 endpoints:
   - `GET /` - Root information
   - `GET /health` - Health check
   - `POST /query` - RAG question answering
   - `POST /upload` - Document upload
   - `POST /search` - Retrieval-only search
   - `DELETE /vectorstore` - Clear all documents
3. ✅ **Error Handling** - Global + endpoint-specific validation
4. ✅ **CORS Support** - Cross-origin request handling
5. ✅ **Auto Documentation** - Swagger UI + ReDoc
6. ✅ **Testing Suite** - Python script for automated testing
7. ✅ **API Examples** - Curl and PowerShell examples
8. ✅ **Docker Support** - Containerization for deployment

### API Endpoints Overview

| Method | Endpoint | Purpose | Auth Required |
|--------|----------|---------|---------------|
| GET | `/` | API information | No |
| GET | `/health` | Health check | No |
| POST | `/query` | Ask questions (RAG) | No |
| POST | `/upload` | Upload documents | No |
| POST | `/search` | Search documents only | No |
| DELETE | `/vectorstore` | Clear all documents | Optional |

### Key Features

- **Production-Ready**: Error handling, validation, CORS
- **Interactive Docs**: Test API in browser at `/docs`
- **Type-Safe**: Pydantic validation on all I/O
- **Containerized**: Docker support for deployment
- **Well-Tested**: Comprehensive test suite included
- **Documented**: Examples for curl, PowerShell, Python

---

## Quick Reference

### Project Commands

```bash
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run main app
python -m app.main

# Run tests
python test_rag.py

# Check graph structure
python -c "from app.rag.graph import rag_graph; print(rag_graph.visualize())"
```

### Key Files

- `app/rag/state.py` - State definitions
- `app/rag/vectorstore.py` - Vector store manager
- `app/rag/nodes.py` - Retrieval and generation nodes
- `app/rag/graph.py` - LangGraph pipeline
- `app/main.py` - Entry point
- `.env` - Configuration

---

**Ready for Part 3: FastAPI Integration!** 🚀
