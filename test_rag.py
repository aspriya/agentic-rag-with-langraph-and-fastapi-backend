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