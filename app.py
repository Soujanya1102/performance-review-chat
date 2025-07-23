import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# Create and save vector store
def get_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embeddings)
        save_path = "faiss_index"
        vector_store.save_local(save_path)

        # Confirm folder and index file creation
        index_file = os.path.join(save_path, "index.faiss")
        if os.path.exists(index_file):
            st.success("✅ FAISS index saved successfully and verified!")
        else:
            st.warning("⚠️ FAISS index folder created, but index.faiss file not found.")
    except Exception as e:
        st.error(f"❌ Error saving FAISS index: {e}")

# Create QA chain with Gemini
def get_conversation_chain():
    prompt_template = """
    You are an assistant helping an employee understand their performance review.
    Answer the question using the context below. If the answer is not in the context, say:
    "The answer is not available in the review document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
 
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index/index.faiss"

    if not os.path.exists(index_path):
        st.error("❌ FAISS index not found. Please upload and process your review document first.")
        return

    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        chain = get_conversation_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("### 🧠 Response:")
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Performance Review Chat", layout="wide")
    st.title("📊 Chat with Your Performance Review")

    user_question = st.text_input("Ask a question about your review:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📄 Upload Review Document")
        pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)
        if st.button("Process Review"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Review processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF document.")

if __name__ == "__main__":
    main()
