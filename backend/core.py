from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from typing import List, Dict, Any

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_openai import OpenAIEmbeddings, ChatOpenAI


INDEX_NAME = "langchain-doc-index"

def run_llm(query:str, chat_history:List[Dict[str,Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    chat = ChatOpenAI(verbose = True, temperature=0, model="gpt-4o-mini")

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")


    retreival_qa_chat_prompt =  hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm=chat, prompt=retreival_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm = chat, retriever = docsearch.as_retriever(), prompt =  rephrase_prompt)



    qa = create_retrieval_chain(
        retriever= history_aware_retriever,
        combine_docs_chain =stuff_documents_chain
    )

    result = qa.invoke({"input": query, "chat_history": chat_history})

    detailed_result = { # This is to debug and view the detailed result
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return detailed_result

if __name__ == "__main__":
    res = run_llm(query = "What is LangChain?")
    print(f" query is {res['query']}")
    print(f" result is {res['result']}")
    print(f" source documents is {res['source_documents']}")

