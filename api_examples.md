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
      "FastAPI is a modern web framework for building APIs with Python.",
      "LangGraph is a library for building stateful agent workflows."
    ],
    "metadata": [
      {"source": "manual", "category": "frameworks"},
      {"source": "manual", "category": "ai"}
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
      "content": "LangGraph is a library...",
      "metadata": {"source": "doc_0", "type": "sample"}
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
      "content": "StateGraph is the main class...",
      "metadata": {"source": "doc_2", "type": "sample"}
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
        "FastAPI is a modern web framework.",
        "LangGraph is a library for AI agents."
    )
    metadata = @(
        @{ source = "test"; category = "framework" }
        @{ source = "test"; category = "ai" }
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
