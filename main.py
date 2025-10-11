from langchain_google_genai import ChatGoogleGenerativeAI as chatbot
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings as hfe
from langchain.chains import RetrievalQA
import os
from embedding import pdf_splitter, load_faiss_index

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
hfe_api_key = os.getenv("HFE_API_KEY")

model = chatbot(model="gemini-2.5-flash", temperature=0, api_key=api_key)
embeddings = hfe(model="google/embeddinggemma-300m",
                 huggingfacehub_api_token=hfe_api_key)


index_path = "faiss_index"
pdf_path = "./docs/example.pdf"

print("Processing PDF and creating index...")
docs = pdf_splitter(pdf_path)
db = load_faiss_index(embeddings, index_path, docs)

print("FAISS index loaded successfully!")

retriever = db.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 10}  
)

qa = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"  
)

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break 
    result = qa.invoke({"query": query})
    
    print("\nAnswer:", result["result"])
    
    if result["source_documents"]:
        print("\nMost relevant source:")
        print(result["source_documents"][0].page_content[:500] + "...")
    
    print("---" * 20)