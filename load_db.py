from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import orjson
from myfunctions import Document

def load_orjson_file(file_path):
    with open(file_path, 'rb') as file:
        data = orjson.loads(file.read())  # Load the JSON data into a Python dictionary or list
    return data

# Load the JSON file
file_path = 'App\DB_files\output_exeo.json'
data = load_orjson_file(file_path)

# Initialize an empty list for documents
documents = []

# Loop through each item in data (assuming it's a list of dictionaries)
for item in data:
    # Append each document, passing 'text' and 'url' from each item
    documents.append(Document(item['text'], {"source": item['url']}))

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embeddings and vector store setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('App\db_faiss_exeo')
