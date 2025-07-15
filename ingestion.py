from dotenv import load_dotenv
import os

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

BATCH_SIZE = 100  # Adjust as needed

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")


    # If its HTML documents in a folder
    # loader = ReadTheDocsLoader(
    #     "file_path/ folder_path", # This is the path to the documentation source files
    #     encoding="utf-8"  # Add this if supported
    # )

    # For pdfs

def ingest_docs():
    pdf_folder = "folder_path" # This is the folder where the pdfs are stored.
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    raw_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        raw_documents.extend(loader.load())

    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        # new_url = new_url.replace("folder_path", "https:/") # This is need if you would like to replace the local file path with a URL
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to the Pinecone vector store")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name="langchian-ingestion-experiment-1", #This is the index name in Pinecone. You will first have to create this index in your Pinecone account
        )

    print("*******Loading to vector store complete!*******")

if __name__ == "__main__":
        ingest_docs()

