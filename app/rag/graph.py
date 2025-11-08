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