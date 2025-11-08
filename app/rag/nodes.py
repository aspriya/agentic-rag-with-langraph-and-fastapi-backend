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