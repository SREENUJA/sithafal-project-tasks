import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
import warnings

# Suppress warnings from Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Load environment variables
load_dotenv()

# Retrieve API Key from environment
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Please set the ANTHROPIC_API_KEY in your environment variables.")
    st.stop()

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


# Function to read PDFs and extract text
def pdf_read(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


# Function to split text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


# Function to create and save vector store
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


# Function to handle the conversational chain and response
def get_conversational_chain(tools, question):
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=API_KEY, verbose=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the context, say you don't know."),
            ("human", "{input}"),
            ("agent", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": question})
    st.write("Reply: ", response['output'])


# Function to handle user input and query processing
def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the PDF.")
        get_conversational_chain(retrieval_chain, user_question)
    except Exception as e:
        st.error(f"Error during processing: {e}")


# Main function to set up the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.header("RAG-based chat with PDFs")

    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the submit & process button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = pdf_read(pdf_docs)
                    if raw_text:
                        text_chunks = get_chunks(raw_text)
                        vector_store(text_chunks)
                        st.success("Processing complete! You can now ask questions.")
                    else:
                        st.warning("No text extracted from the uploaded PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")


# Ensure main function is executed
if _name_ == "_main_":  # Corrected the main guard
    main()
