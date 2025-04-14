from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from openai import OpenAI
from typing import List, Optional, Dict
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

# Custom Embeddings class that implements LangChain's Embeddings interface
class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings using OpenAI-compatible API."""
    
    def __init__(
        self,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using DashScope."""
        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch: {e}")
                # Return empty embeddings for failed batch
                embeddings.extend([[0.0] * self.dimensions] * len(batch))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using DashScope."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Return empty embedding on failure
            return [0.0] * self.dimensions

# Initialize the custom embeddings class
embeddings = DashScopeEmbeddings()

# Initialize ChatOpenAI for LLM interactions
llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

# Example usage with different document types
def create_sample_vectorstore():
    """Create and save a sample vector store with different document types."""
    # Create Document objects with different types and metadata
    documents = [
        Document(
            page_content="Elonmusk adalah seorang pengusaha teknologi dan pendiri beberapa perusahaan terkenal.",
            metadata={"source": "biography", "type": "person", "id": "elon_musk_bio"}
        ),
        Document(
            page_content="Tesla adalah perusahaan mobil listrik yang didirikan oleh Elon Musk. Perusahaan ini terkenal dengan inovasi teknologi dan desain yang futuristik.",
            metadata={"source": "company_profile", "type": "company", "id": "tesla_profile"}
        ),
        Document(
            page_content="SpaceX adalah perusahaan eksplorasi luar angkasa yang didirikan oleh Elon Musk dengan tujuan mengurangi biaya transportasi luar angkasa dan memungkinkan kolonisasi Mars.",
            metadata={"source": "company_profile", "type": "company", "id": "spacex_profile"}
        ),
        Document(
            page_content="Laporan keuangan Tesla menunjukkan peningkatan pendapatan sebesar 30% pada kuartal terakhir, didorong oleh penjualan Model Y yang kuat.",
            metadata={"source": "financial_report", "type": "report", "id": "tesla_q4_2024"}
        ),
        Document(
            page_content="Peluncuran roket Starship oleh SpaceX mencapai orbit dengan sukses pada misi terakhir, menandai tonggak penting dalam program luar angkasa perusahaan.",
            metadata={"source": "news", "type": "event", "id": "spacex_launch_2024"}
        )
    ]
    
    # Create vector store from documents
    vstore = FAISS.from_documents(documents, embeddings)
    retriever = vstore.as_retriever()
    
    # Save the vector store locally
    vstore.save_local("faiss_store")
    
    print(f"Added {len(documents)} documents to the vector store.")
    
    return vstore, retriever

def query_vectorstore(retriever, query_text: str) -> str:
    """Query the vector store and generate an answer using RAG."""
    # Create a prompt template for retrieval that includes metadata
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Include relevant information from the document metadata if available.

    Context:
    {context}

    Question: {input}

    Answer:
    """)
    
    # Create a retrieval chain using the LCEL pattern
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Execute the chain
    answer = rag_chain.invoke(query_text)
    
    return answer

def filter_by_metadata(vstore, metadata_filter):
    """Create a retriever that filters by metadata."""
    return vstore.as_retriever(
        search_kwargs={"k": 3, "filter": metadata_filter}
    )

if __name__ == "__main__":
    try:
        # Create and save a sample vector store
        print("Creating sample vector store...")
        vstore, retriever = create_sample_vectorstore()
        
        # Test multiple queries to demonstrate retrieval across different documents
        queries = [
            "Siapa Elon Musk?",
            "Apa itu Tesla?",
            "Apa tujuan SpaceX?"
        ]
        
        for query in queries:
            print("\n" + "="*50)
            print(f"Query: {query}")
            print("="*50)
            
            # Retrieve documents for debugging
            retrieved_docs = retriever.invoke(query)
            
            print("\nRetrieved documents:")
            for i, doc in enumerate(retrieved_docs):
                print(f"{i+1}. [{doc.metadata.get('type', 'unknown')}] {doc.page_content}")
                print(f"   Source: {doc.metadata.get('source', 'unknown')}, ID: {doc.metadata.get('id', 'unknown')}")
            
            # Get answer
            print("\nGenerating answer...")
            answer = query_vectorstore(retriever, query)
            print(f"\nAnswer: {answer}")
        
        # Demonstrate metadata filtering
        print("\n" + "="*50)
        print("Filtering by metadata (only company documents)")
        print("="*50)
        
        # Create a filtered retriever
        company_retriever = filter_by_metadata(vstore, {"type": "company"})
        query = "Berikan informasi tentang perusahaan Elon Musk"
        
        # Retrieve only company documents
        filtered_docs = company_retriever.invoke(query)
        
        print("\nRetrieved company documents:")
        for i, doc in enumerate(filtered_docs):
            print(f"{i+1}. [{doc.metadata.get('type', 'unknown')}] {doc.page_content}")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}, ID: {doc.metadata.get('id', 'unknown')}")
        
        # Get answer using filtered documents
        answer = query_vectorstore(company_retriever, query)
        print(f"\nAnswer (based on company documents only): {answer}")
        
        # Load the saved vector store
        print("\n" + "="*50)
        print("Loading saved vector store...")
        loaded_vstore = FAISS.load_local("faiss_store", embeddings)
        print(f"Loaded vector store successfully!")
        print("="*50)
    except Exception as e:
        print(f"Error: {e}")
