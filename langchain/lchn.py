import token
from langchain_core import embeddings
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()


# client = ChatNVIDIA(
#   model="mistralai/mixtral-8x7b-instruct-v0.1",
#   api_key=os.getenv("NVIDIA_API_KEY"), 
#   temperature=0.5,
#   top_p=0.7,
#   max_tokens=1024,
#   stream=True
# )

# for chunk in client.stream([{"role":"system","content":"You are a helpful assistant in Indonesian Law."},{"role":"user","content":"apa saja syarat untuk membuat ktp?"}]): 
#   print(chunk.content, end="")

chat = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)



# Load IndoBERT model using LangChain's HuggingFaceEmbeddings
# indobert_embeddings = HuggingFaceEmbeddings(
#     model_name="",
#     model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
#     encode_kwargs={"normalize_embeddings": True}  # Equivalent to normalize_embeddings=True
# )

ollama = OllamaEmbeddings(
    model="llama3.1",
    num_gpu=1,
    verbose=True
)

print(ollama)

# Contoh dokumen hukum
dokumen = "Pasal 1 Ayat 1: Setiap orang memiliki hak asasi yang diakui oleh hukum."

# Generate embeddings using LangChain
vector = ollama.embed_query(dokumen)

# Print vector shape
print("Vector Shape:", (len(vector),))  # Should be (768,)

# Example of how to embed multiple documents at once with LangChain
dokumen_list = [
    "Pasal 1 Ayat 1: Setiap orang memiliki hak asasi yang diakui oleh hukum.",
    "Pasal 2 Ayat 3: Negara menjamin hak setiap warga untuk mendapatkan pendidikan.",
    "Pasal 5 Ayat 2: Setiap warga negara berhak atas pekerjaan dan penghidupan yang layak."
]

# Embed multiple documents
# vectors = ollama.embed_documents(dokumen_list)
# print(f"Number of documents embedded: {len(vectors)}")
# print(f"Each vector shape: {len(vectors[0])}")  # Should be 768

# vstore = FAISS.from_texts(dokumen_list, ollama)
# retriever = vstore.as_retriever()

# print(retriever.invoke("Apa bunyi Pasal 2 ayat 3?"))
