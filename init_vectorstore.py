"""
Initialize vector store with sample documents.
"""

from app.rag.vectorstore import VectorStoreManager

# Sample documents
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

# Initialize and upload
vm = VectorStoreManager()
result = vm.upload_documents(sample_documents, metadata)

print(f"\n✅ Uploaded {result['documents_added']} documents")
print(f"Total documents in store: {result['total_documents']}\n")
