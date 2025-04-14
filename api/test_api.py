import httpx
import asyncio
import json

async def test_completion_api():
    """
    Test the completion API endpoint
    """
    url = "http://localhost:8000/api/completion"
    
    # Test data that matches the format from server.js
    data = {
        "input": {
            "prompt": "What are the basic legal rights in Indonesia?"
        },
        "parameters": {},
        "debug": {}
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"Sending request to {url}...")
            print(f"Request data: {json.dumps(data, indent=2)}")
            
            response = await client.post(
                url,
                json=data,
                timeout=30.0
            )
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                print("Success!")
                try:
                    print(json.dumps(response.json(), indent=2))
                except Exception as e:
                    print(f"Could not parse JSON response: {str(e)}")
                    print(f"Raw response: {response.text[:500]}...")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_completion_api())
