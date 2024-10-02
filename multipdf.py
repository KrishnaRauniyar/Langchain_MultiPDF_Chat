import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

### Function to read all pages in PDFs and append text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

### Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

### Create and save FAISS vector store
def get_vector_store(text_chunks):
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

### Load conversational chain with custom prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context. If the answer is not available in the context, simply respond with 'Answer not available in the context of given pdfs.'
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

### Handle user input and query the PDF content
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    
    except RuntimeError as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

def main():
    # Page layout and configuration
    st.set_page_config(page_title="Chat with Multiple PDF AI", page_icon="📄", layout="wide")
    
    # Header section with a styled title and description
    st.title("💁 Chat with Multiple PDF using Gemini 📄")
    st.write("""
        **Upload your PDF files**, let the app process them, and then ask questions. Our AI will extract and analyze the text for you! 
    """)
    
    st.markdown("---")
    
    # Create two columns: one for PDF upload and another for the conversation interface
    col1, col2 = st.columns(2)
    
    # Sidebar for file uploading
    with col1:
        st.header("📂 Upload PDF Files")
        st.write("You can upload multiple PDFs at once.")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        
        if pdf_docs:
            if st.button("Process Files"):
                with st.spinner("Reading and processing PDFs..."):
                    start_time = time.time()
                    
                    # Read and process PDF files
                    raw_text = get_pdf_text(pdf_docs)
                    st.info(f"Extracted {len(raw_text)} characters from the PDF(s).")
                    
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.info(f"Text split into {len(text_chunks)} chunks for processing.")
                    
                    # Create and save vector store
                    get_vector_store(text_chunks)
                    st.success(f"PDF processing completed in {round(time.time() - start_time, 2)} seconds!")
                    st.balloons()
        else:
            st.warning("Please upload PDF files to process.")

    # Question and Answer Section
    with col2:
        st.header("💬 Ask Questions")
        user_question = st.text_input("Type your question here:")
        
        if user_question:
            with st.spinner("Generating response..."):
                response = user_input(user_question)
                if response:
                    st.success("Here's your answer:")
                    st.write(response)
                else:
                    st.error("No answer found or an error occurred.")
        else:
            st.write("Type a question after uploading and processing PDFs.")
    
    # Display processing steps in an expander for more user transparency
    with st.expander("📋 How It Works"):
        st.write("""
            1. **Upload PDFs**: Upload your PDFs in the left section.
            2. **Processing**: The app extracts text from the PDFs and divides it into manageable chunks.
            3. **Vectorization**: The chunks are converted into vector embeddings for fast search and retrieval.
            4. **Ask Questions**: Enter your question, and the app will find relevant information and provide an answer.
        """)

if __name__ == "__main__":
    main()
