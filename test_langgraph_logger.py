import os
import sys

# Import directly from the file
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the initialize_scraper function directly
try:
    # First, try to import directly
    from langchain.langgraph import initialize_scraper
    print("Successfully imported from langchain.langgraph")
except ImportError:
    # If that fails, try importing the module directly
    print("Direct import failed, trying alternative approach")
    import importlib.util
    spec = importlib.util.spec_from_file_location("langgraph", "langchain/langgraph.py")
    langgraph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(langgraph)
    initialize_scraper = langgraph.initialize_scraper
    print("Successfully imported using importlib")

# Set up environment variables if needed
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key")

def test_initialize_scraper():
    print("Testing initialize_scraper function...")
    try:
        scraper = initialize_scraper()
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
