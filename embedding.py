from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

def pdf_splitter(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

def load_faiss_index(embeddings, index_path="faiss_index", docs=None):
    # Check if index already exists by checking for both required files
    index_files_exist = (
        os.path.exists(os.path.join(index_path, "index.faiss")) and
        os.path.exists(os.path.join(index_path, "index.pkl"))
    )
    
    if index_files_exist:
        print("Loading existing FAISS index...")
        try:
            db = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return db
        except Exception as e:
            print(f"Error loading index: {e}. Creating new index...")
    
    if docs is None:
        raise ValueError("Documents are required to create a new index.")
    
    print("Creating new FAISS index...")
    # Generate embeddings and create index
    db = FAISS.from_documents(docs, embeddings)
    # Create directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)
    print("FAISS index created and saved locally.")
    return db