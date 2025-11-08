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
            "FastAPI is a modern, fast web framework for building APIs with Python.",
            "FastAPI is based on standard Python type hints and provides automatic API documentation.",
            "Uvicorn is an ASGI server that runs FastAPI applications."
        ],
        "metadata": [
            {"source": "test", "category": "framework"},
            {"source": "test", "category": "features"},
            {"source": "test", "category": "deployment"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/upload", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_search_documents():
    """Test search endpoint."""
    print_section("TEST 3: Search Documents")
    
    payload = {
        "question": "What is FastAPI?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_query_rag():
    """Test RAG query endpoint."""
    print_section("TEST 4: RAG Query")
    
    payload = {
        "question": "What is FastAPI and how does it relate to Uvicorn?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"Answer: {result['answer']}\n")
    print(f"Sources ({len(result['sources'])} documents):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  [{i}] {source['content'][:100]}...")
        print(f"      Metadata: {source['metadata']}")


def test_invalid_request():
    """Test error handling with invalid request."""
    print_section("TEST 5: Invalid Request (Error Handling)")
    
    payload = {
        "question": "",  # Empty question should fail
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_multiple_queries():
    """Test multiple queries in sequence."""
    print_section("TEST 6: Multiple Queries")
    
    questions = [
        "What is StateGraph in LangGraph?",
        "Does LangGraph support checkpointing?",
        "How can I visualize a LangGraph workflow?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuery {i}: {question}")
        payload = {"question": question, "top_k": 2}
        response = requests.post(f"{BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:150]}...")
        else:
            print(f"Error: {response.status_code}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  RAG BACKEND API - TEST SUITE")
    print("="*80)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API is running: uvicorn app.main:app --reload\n")
    
    try:
        # Test root endpoint first
        response = requests.get(BASE_URL)
        print("✅ API is reachable!\n")
        
        # Run all tests
        test_health()
        test_upload_documents()
        test_search_documents()
        test_query_rag()
        test_invalid_request()
        test_multiple_queries()
        
        print("\n" + "="*80)
        print("  ✅ ALL TESTS COMPLETED!")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API!")
        print("Please start the API first: uvicorn app.main:app --reload\n")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()
