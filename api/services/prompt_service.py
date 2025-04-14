import os
from dotenv import load_dotenv
import httpx
from fastapi import HTTPException

# Load environment variables
load_dotenv()

class PromptService:
    def __init__(self):
        # Try to get API key from environment, if not available, use the one from server.js
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-0054022384a64f03abfdbcae8c001cbb")
        self.api_url = "https://dashscope-intl.aliyuncs.com/api/v1/apps/172f5d2e8d1b4b9da47dada83dcb7f19/completion"
        
    async def process_prompt(self, prompt: str):
        """
        Process a prompt using the Dashscope API
        """
        if not self.api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "input": {
                            "prompt": prompt
                        },
                        "parameters": {},
                        "debug": {}
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-SSE": "enable",
                        "Accept": "text/event-stream"
                    },
                    timeout=30.0
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e.response.json()))
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
