import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize models
embeddings = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0.3,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

paraphrase_llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

def log_message(level, message):
    """Simple logging function"""
    levels = {
        "info": "\033[94m[INFO]\033[0m",
        "success": "\033[92m[SUCCESS]\033[0m",
        "warning": "\033[93m[WARNING]\033[0m",
        "error": "\033[91m[ERROR]\033[0m"
    }
    print(f"{levels.get(level, '[LOG]')} {message}")

def calculate_query_similarity(query1, query2):
    """Calculate semantic similarity between two queries using embeddings."""
    try:
        # Get embeddings for both queries
        embedding1 = embeddings.embed_query(query1)
        embedding2 = embeddings.embed_query(query2)
        
        # Convert to numpy arrays for calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        log_message("info", f"Query similarity: {similarity:.4f}")
        return float(similarity)
    except Exception as e:
        log_message("error", f"Error calculating query similarity: {str(e)}")
        return 0.0  # Default to 0 on error

def paraphrase_query(query_text, style="standard", min_similarity=0.8):
    """Paraphrase a user query with similarity check."""
    style_descriptions = {
        "standard": "Maintain the same level of formality and tone as the original.",
        "expanded": "Expand the query with related terms that might improve search results.",
        "precise": "Make the query more specific and focused on the core information need."
    }
    
    style_desc = style_descriptions.get(style.lower(), style_descriptions["standard"])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at reformulating search queries to improve information retrieval results.
    Paraphrase the following query while STRICTLY preserving its original intent, meaning, and all key terms.
    
    IMPORTANT GUIDELINES:
    1. Preserve ALL named entities, technical terms, and specific concepts from the original query
    2. Do not introduce new concepts that weren't in the original query
    3. Ensure your paraphrase would retrieve the same information as the original
    4. Keep the query focused on the same topic and information need
    
    Style instructions: {style_desc}
    
    Original query:
    {input_text}
    
    Paraphrased query:
    """)
    
    try:
        chain = prompt | paraphrase_llm | StrOutputParser()
        result = chain.invoke({
            "input_text": query_text,
            "style_desc": style_desc
        })
        
        # Check similarity between original and paraphrased query
        similarity = calculate_query_similarity(query_text, result)
        
        # If similarity is below threshold, use original query
        if similarity < min_similarity:
            log_message("warning", f"Paraphrased query similarity ({similarity:.4f}) below threshold ({min_similarity}). Using original query.")
            return query_text
            
        log_message("success", f"Query paraphrased in {style} style with similarity {similarity:.4f}")
        return result
    except Exception as e:
        log_message("error", f"Error paraphrasing query: {str(e)}")
        return query_text  # Return original query if paraphrasing fails

def query_vectorstore(retriever, query):
    """Query the vector store and generate an answer."""
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
        if not docs:
            return {
                "has_answer": False,
                "answer": "I don't have enough information to answer this question",
                "sources": []
            }
        
        # Create a prompt for generating an answer
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based ONLY on the provided context.
        If the context doesn't contain enough information to answer the question, 
        respond with "I don't have enough information to answer this question".
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """)
        
        # Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate an answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query})
        
        # Check if the answer indicates insufficient information
        has_answer = "I don't have enough information" not in answer
        
        # Format the result
        result = {
            "has_answer": has_answer,
            "answer": answer,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        }
        
        return result
    except Exception as e:
        return {
            "has_answer": False,
            "error": str(e),
            "sources": []
        }

def compare_retrieval_results(retriever, original_query, paraphrase_style="expanded"):
    """Compare retrieval results between original and paraphrased queries with fallback."""
    log_message("info", f"Comparing retrieval results for original and paraphrased query ({paraphrase_style} style)")
    
    # Get paraphrased query
    paraphrased_query = paraphrase_query(original_query, paraphrase_style)
    
    # Retrieve documents for original query
    log_message("info", f"Retrieving documents for original query: {original_query}")
    original_docs = retriever.invoke(original_query)
    
    # Get answer for original query
    log_message("info", f"Generating answer for original query")
    original_answer = query_vectorstore(retriever, original_query)
    
    # Retrieve documents for paraphrased query
    log_message("info", f"Retrieving documents for paraphrased query: {paraphrased_query}")
    paraphrased_docs = retriever.invoke(paraphrased_query)
    
    # Get answer for paraphrased query
    log_message("info", f"Generating answer for paraphrased query")
    paraphrased_answer = query_vectorstore(retriever, paraphrased_query)
    
    # Check if paraphrased query has an answer
    has_paraphrased_answer = paraphrased_answer.get("has_answer", True)
    
    # If paraphrased query doesn't have an answer but original does, use original as fallback
    if not has_paraphrased_answer and original_answer.get("has_answer", False):
        log_message("warning", "Paraphrased query didn't yield results, using original query results as fallback")
        paraphrased_answer = {
            "has_answer": True,
            "answer": f"[FALLBACK TO ORIGINAL QUERY] {original_answer.get('answer', '')}",
            "sources": original_answer.get("sources", []),
            "used_fallback": True
        }
    
    # Format results
    results = {
        "original_query": original_query,
        "paraphrased_query": paraphrased_query,
        "paraphrase_style": paraphrase_style,
        "original_results": {
            "answer": original_answer
        },
        "paraphrased_results": {
            "answer": paraphrased_answer
        },
        "used_fallback": paraphrased_answer.get("used_fallback", False)
    }
    
    return results

def create_test_vectorstore():
    """Create a test vector store with sample documents."""
    # Sample documents about financial results
    financial_docs = [
        Document(
            page_content="In Q1 2023, the company reported revenue of $10.5 million, with a profit of $2.3 million. This represents a 15% increase in revenue compared to Q1 2022.",
            metadata={"source": "financial_report", "id": "fin001", "type": "report"}
        ),
        Document(
            page_content="The technology sector has grown by 8% in 2023, with AI applications leading the growth.",
            metadata={"source": "market_analysis", "id": "ma001", "type": "report"}
        ),
        Document(
            page_content="John Smith is the CEO of the company since 2020. He previously worked at Tech Giants Inc.",
            metadata={"source": "company_profile", "id": "cp001", "type": "person"}
        ),
        Document(
            page_content="The company launched three new products in Q4 2022: ProductA, ProductB, and ProductC.",
            metadata={"source": "product_catalog", "id": "pc001", "type": "report"}
        ),
        Document(
            page_content="Recent security audits revealed no critical vulnerabilities in the company's systems.",
            metadata={"source": "security_report", "id": "sr001", "type": "report"}
        )
    ]
    
    # Create vector store
    vectorstore = FAISS.from_documents(financial_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def test_fallback_mechanism():
    """Test the fallback mechanism with a sample vector store."""
    print("\n" + "="*80)
    print("TESTING FALLBACK MECHANISM FOR NO-ANSWER SCENARIOS".center(80))
    print("="*80 + "\n")
    
    # Create test vector store
    retriever = create_test_vectorstore()
    
    # Test queries
    test_queries = [
        {
            "original": "What were the financial results in Q1 2023?",
            "poor_paraphrase": "What were the accounting outcomes for January to March 2023 fiscal period?",
            "description": "Query with financial information (should have answer)"
        },
        {
            "original": "What cybersecurity measures were implemented in 2023?",
            "poor_paraphrase": "What digital protection protocols were established during 2023?",
            "description": "Query with limited information (might not have answer)"
        }
    ]
    
    for test in test_queries:
        print(f"\n--- Testing: {test['description']} ---\n")
        
        # Test original query
        print(f"Original Query: \"{test['original']}\"")
        original_result = query_vectorstore(retriever, test['original'])
        print(f"Has Answer: {original_result['has_answer']}")
        print(f"Answer: {original_result['answer']}\n")
        
        # Test poor paraphrase
        print(f"Poor Paraphrase: \"{test['poor_paraphrase']}\"")
        paraphrase_result = query_vectorstore(retriever, test['poor_paraphrase'])
        print(f"Has Answer: {paraphrase_result['has_answer']}")
        print(f"Answer: {paraphrase_result['answer']}\n")
        
        # Test comparison with fallback
        print("Testing comparison with fallback:")
        comparison = compare_retrieval_results(retriever, test['original'])
        
        print(f"Original Query: \"{comparison['original_query']}\"")
        print(f"Paraphrased Query: \"{comparison['paraphrased_query']}\"")
        print(f"Used Fallback: {comparison['used_fallback']}")
        
        if comparison['used_fallback']:
            print(f"Fallback Answer: {comparison['paraphrased_results']['answer']['answer']}")
        else:
            print(f"Paraphrased Answer: {comparison['paraphrased_results']['answer']['answer']}")
        
        print("\n" + "-"*60)

if __name__ == "__main__":
    test_fallback_mechanism()
