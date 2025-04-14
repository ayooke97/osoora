# Osoora FastAPI Service

This is the FastAPI backend for the Osoora legal assistance service.

## Features

- RESTful API for processing legal queries
- Integration with Dashscope API for completions
- Structured for scalability with routers, services, and models

## Project Structure

```
api/
├── main.py              # Main application entry point
├── .env.example         # Example environment variables
├── routers/             # API route definitions
│   └── completion.py    # Completion API endpoints
├── services/            # Business logic
│   └── prompt_service.py # Service for processing prompts
├── models/              # Data models (to be added)
├── utils/               # Utility functions (to be added)
└── tests/               # Test cases (to be added)
```

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```

2. Install dependencies (if not already installed):
   ```
   pip install -r requirements.txt
   ```

3. Run the server:
   ```
   python main.py
   ```
   
   Or using uvicorn directly:
   ```
   uvicorn api.main:app --reload
   ```

## API Endpoints

### POST /api/completion

Process a prompt and return completion results.

**Request Body:**
```json
{
  "input": {
    "prompt": "Your prompt text here"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "result": {
    // Response data from the completion API
  }
}
```

## Integration with Existing Frontend

To integrate with the existing Node.js server, modify the axios request in `server.js` to point to this FastAPI service instead of directly to the Dashscope API.
