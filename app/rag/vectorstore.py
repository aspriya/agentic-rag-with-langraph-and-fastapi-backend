import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    """Manages FAISS vector store for document storage and retrieval."""
    
    def __init__(
        self,
        vectorstore_path: Optional[str] = None,
        embedding_model: Optional[str] = None
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
    
    def upload_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> dict:
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
    
    def get_retriever(self, top_k: Optional[int] = None):
        """
        Get a retriever for the vector store.
        
        Args:
            top_k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        k = top_k or int(os.getenv("TOP_K_DOCUMENTS", "3"))
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
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