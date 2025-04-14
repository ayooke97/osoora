import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Vector Store API",
    description="API for querying and managing vector stores with LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data for mock responses
sample_responses = {
    "person": {
        "Michael Brown": "Michael Brown is a project manager with experience in agile methodologies.",
        "John Smith": "John Smith is a software engineer with 10 years of experience in Python and JavaScript.",
        "Sarah Johnson": "Sarah Johnson is a data scientist specializing in machine learning and AI applications."
    },
    "company": {
        "Acme Corp": "Acme Corp is a technology company founded in 2005, specializing in cloud solutions.",
        "TechGiant": "TechGiant Inc. is a global leader in AI and machine learning technologies."
    }
}

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>LangChain Vector Store API</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2 {
                    color: #2c3e50;
                }
                h1 {
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                .endpoint {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }
                .method {
                    font-weight: bold;
                    color: #2980b9;
                }
                pre {
                    background-color: #f1f1f1;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <h1>LangChain Vector Store API</h1>
            <p>This API provides endpoints for querying and managing vector stores with LangChain.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> /health - Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">GET</span> /query - Query the vector store</p>
                <pre>
/query?query=Who is Michael Brown?&filter_type=person
                </pre>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> /query - Query the vector store</p>
                <pre>
{
  "query": "Who is Michael Brown?",
  "filter_type": "person"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> /synonyms - Extract document-related synonyms</p>
                <pre>
{
  "text": "Find employees and organizations related to AI"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> /paraphrase - Paraphrase text</p>
                <pre>
{
  "text": "The quick brown fox jumps over the lazy dog",
  "style": "creative"
}
                </pre>
            </div>
            
            <p>For full API documentation, visit <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a>.</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/query")
async def query_endpoint_get(query: str, filter_type: Optional[str] = None):
    """Query the vector store using GET method."""
    return process_query(query, filter_type)

@app.post("/query")
async def query_endpoint_post(request: Request):
    """Query the vector store using POST method."""
    # Parse JSON request body
    data = await request.json()
    query = data.get("query", "")
    filter_type = data.get("filter_type")
    
    return process_query(query, filter_type)

@app.post("/synonyms")
async def synonyms_endpoint(request: Request):
    """Extract document-related synonyms from text."""
    # Parse JSON request body
    data = await request.json()
    text = data.get("text", "")
    
    # Mock synonym extraction
    synonyms = {}
    
    # Check for person-related terms
    person_terms = ["employee", "staff", "individual", "person", "people"]
    if any(term in text.lower() for term in person_terms):
        synonyms["person"] = [term for term in person_terms if term in text.lower()]
    
    # Check for company-related terms
    company_terms = ["organization", "firm", "business", "company", "corporation"]
    if any(term in text.lower() for term in company_terms):
        synonyms["company"] = [term for term in company_terms if term in text.lower()]
    
    return {"synonyms": synonyms}

@app.post("/paraphrase")
async def paraphrase_endpoint(request: Request):
    """Paraphrase text."""
    # Parse JSON request body
    data = await request.json()
    text = data.get("text", "")
    style = data.get("style", "standard")
    
    # Mock paraphrasing based on style
    paraphrased = text
    
    if style == "formal":
        paraphrased = text.replace("quick", "expeditious").replace("jumps", "traverses")
    elif style == "simple":
        paraphrased = text.replace("quick", "fast").replace("jumps", "jumps")
    elif style == "creative":
        paraphrased = text.replace("quick", "swift").replace("jumps", "leaps")
    
    return {
        "original": text,
        "paraphrased": paraphrased,
        "style": style
    }

def process_query(query: str, filter_type: Optional[str] = None):
    """Process query and return results."""
    # Default response
    default_response = "I don't have enough information to answer this question."
    
    # Check if query contains any of our sample names
    answer = default_response
    sources = []
    
    # Determine which category to search in
    categories = [filter_type] if filter_type and filter_type in sample_responses else sample_responses.keys()
    
    for category in categories:
        for name, info in sample_responses[category].items():
            if name.lower() in query.lower():
                answer = info
                sources.append({
                    "content": info,
                    "metadata": {"type": category, "source": f"{category}_database", "id": f"{name.lower().replace(' ', '_')}"}
                })
                break
        if sources:  # If we found a match, stop searching
            break
    
    # Format the response
    response = {
        "query": query,
        "answer": answer,
        "sources": sources
    }
    
    return response

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
