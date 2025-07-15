from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

BATCH_SIZE = 100  # Adjust as needed

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

def ingest_docs():

    loader = ReadTheDocsLoader(
        "C:/Users/KULUS_JP/Desktop/Python stuff/sublime text/VS Studio learnings/Lang chain/LangChain_Documentation_Helper/documentation-helper/langchain-docs/api.python.langchain.com/en/latest",
        encoding="utf-8"  # Add this if supported
    )

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("C:/Users/KULUS_JP/Desktop/Python stuff/sublime text/VS Studio learnings/Lang chain/LangChain_Documentation_Helper/documentation-helper/langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to the Pinecone vector store")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name="langchain-doc-index",
        )

    print("*******Loading to vector store complete!*******")

if __name__ == "__main__":
        ingest_docs()

