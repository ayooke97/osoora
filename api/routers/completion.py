from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ..services.prompt_service import PromptService

router = APIRouter(
    prefix="/api",
    tags=["completion"]
)

class Input(BaseModel):
    prompt: str

class PromptInput(BaseModel):
    input: Input
    parameters: Optional[Dict[str, Any]] = {}
    debug: Optional[Dict[str, Any]] = {}

class PromptResponse(BaseModel):
    status: str
    result: dict

# Dependency to get the prompt service
def get_prompt_service():
    return PromptService()

@router.post("/completion", response_model=PromptResponse)
async def process_prompt(
    input_data: PromptInput,
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Process a prompt and return the completion result
    """
    try:
        # Format the prompt to request Markdown responses
        formatted_prompt = f"Please format your response using Markdown with proper headings, lists, code blocks, and other formatting where appropriate. Here's the user's message:\n\n{input_data.input.prompt}"
        
        # Process the prompt
        result = await prompt_service.process_prompt(formatted_prompt)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
