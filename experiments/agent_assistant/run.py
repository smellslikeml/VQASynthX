# This script requires: streamlit, langchain, langchain-openai, faiss-cpu, sentence-transformers
# Run from the repository root: `streamlit run experiments/agent_assistant/run.py`

import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# --- Configuration ---
README_PATH = "./README.md"
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def check_api_key():
    """Checks for the OpenAI API key and prompts the user if not found."""
    if "OPENAI_API_KEY" not in os.environ:
        try:
            # Check if it's stored in streamlit secrets
            st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            # Prompt user in the sidebar
            st.sidebar.warning("OpenAI API Key not found. Please enter it below.")
            api_key = st.sidebar.text_input(
                "OpenAI API Key", type="password", key="api_key_input"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.openai_api_key = api_key
                st.rerun()  # Rerun to hide the input and proceed
            st.stop()
    else:
        st.session_state.openai_api_key = os.environ["OPENAI_API_KEY"]


@st.cache_resource
def create_retriever():
    """Creates a FAISS vector store and retriever from the VQASynth README."""
    if not os.path.exists(README_PATH):
        st.error(f"Error: Could not find {README_PATH}.")
        st.error(
            "Please run this script from the root of the 'experimental-vqasynth' repository using `streamlit run experiments/agent_assistant/run.py`."
        )
        st.stop()

    loader = TextLoader(README_PATH, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore.as_retriever()


def main():
    """Main function to run the VQASynth Pipeline Assistant."""
    st.set_page_config(
        page_title="VQASynth Pipeline Assistant",
        page_icon="🤖",
    )
    st.title("🤖 VQASynth Pipeline Assistant")
    st.markdown(
        "This is an experimental AI assistant to help you understand the VQASynth pipeline. "
        "It uses Retrieval-Augmented Generation (RAG) on the project's `README.md` file. "
        "Inspired by the domain-specific agent in [santiagocasas/clapp](https://github.com/santiagocasas/clapp)."
    )

    check_api_key()

    retriever = create_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=0.3,
            openai_api_key=st.session_state.openai_api_key,
        ),
        chain_type="stuff",
        retriever=retriever,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("e.g., What is VQASynth?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(prompt)
                result = response.get(
                    "result",
                    "Sorry, I encountered an issue and could not get a result.",
                )
                st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
