from typing import Set

from backend.core import run_llm
import streamlit as st

st.header( "LangChain Documentation Helper")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(sources_urls: Set[str]) -> str:
    if not sources_urls:
        return "No sources found."
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"i+1. {source}\n"
    return sources_string



if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(
            query = prompt, 
            chat_history = st.session_state["chat_history"]
        )
        sources = set (
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        ) # This returns a set of unique sources the set output is in the following formart: sources = {"file1.pdf", "file2.pdf", "file3.pdf"}

        formatted_response = (f"{generated_response['result']}\n\n {create_sources_string(sources)}")

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", formatted_response))

if st.session_state["chat_answers_history"]:
    for generated_response, user_prompt in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"]
    ):
       st.chat_message("user").write(user_prompt)
       st.chat_message("assistant").write(generated_response)