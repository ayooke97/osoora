import os
import sys
import importlib.util

# Add the langchain directory to the path so imports work correctly
sys.path.insert(0, os.path.abspath('langchain'))

# Load the langgraph.py module directly
print("Loading langgraph.py directly...")
spec = importlib.util.spec_from_file_location("langgraph", "langchain/langgraph.py")
langgraph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(langgraph)

# Set up environment variables if needed
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key")

def test_initialize_scraper():
    print("Testing initialize_scraper function...")
    try:
        scraper = langgraph.initialize_scraper()
        print("✅ Scraper initialized successfully")
        return scraper
    except Exception as e:
        print(f"❌ Error initializing scraper: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== Testing Logging in langgraph.py ===\n")
    
    # Test initialize_scraper
    scraper = test_initialize_scraper()
    
    print("\n=== Testing Complete ===")
