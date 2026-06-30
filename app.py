import os
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from sentence_transformers import CrossEncoder

# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="RAG + Reranker + MLflow", layout="wide")
st.title("📄 RAG + Reranker + Observability + MLflow")

# -------------------------------------------------
# Temp folder
# -------------------------------------------------
os.makedirs("temp", exist_ok=True)

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.reranker = None
    st.session_state.llm = None
    st.session_state.logs = []

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("⚙️ Chunk Settings")

chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 100, 50)

# -------------------------------------------------
# Display config
# -------------------------------------------------
st.subheader("🔧 Current Configuration")
col1, col2 = st.columns(2)
col1.metric("Chunk Size", chunk_size)
col2.metric("Chunk Overlap", chunk_overlap)

st.markdown("---")

# -------------------------------------------------
# Process PDF
# -------------------------------------------------
def process_pdf(pdf_file, chunk_size, chunk_overlap):
    pdf_path = f"temp/{pdf_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    return retriever, reranker, llm

# -------------------------------------------------
# Rerank
# -------------------------------------------------
def rerank_documents(query, docs, reranker, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded_pdf = st.file_uploader("📤 Upload PDF", type=["pdf"])

if uploaded_pdf:
    if st.button("📥 Process PDF"):
        with st.spinner("Processing..."):
            retriever, reranker, llm = process_pdf(
                uploaded_pdf, chunk_size, chunk_overlap
            )

            st.session_state.retriever = retriever
            st.session_state.reranker = reranker
            st.session_state.llm = llm

            st.success("✅ PDF processed!")

# -------------------------------------------------
# Ask Question + MLflow
# -------------------------------------------------
question = st.text_input("💬 Ask a question")

if question:
    if st.session_state.retriever is None:
        st.warning("Upload PDF first")
    else:
        with mlflow.start_run():

            start_time = time.time()

            retriever = st.session_state.retriever
            reranker = st.session_state.reranker
            llm = st.session_state.llm

            # 🔍 Retrieval
            docs = retriever.invoke(question)

            # 🎯 Reranking
            top_docs = rerank_documents(question, docs, reranker)

            # 📄 Context
            context = "\n\n".join([doc.page_content for doc in top_docs])

            # 🧠 Prompt
            prompt = f"""
You are a helpful assistant.

Answer based on the context below.
Infer meaning even if wording differs.

Only say "Not found in document" if completely unrelated.

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            end_time = time.time()
            latency = round(end_time - start_time, 2)

            # ---------------- MLflow Logging ----------------
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)

            mlflow.log_metric("latency", latency)
            mlflow.log_metric("retrieved_docs", len(docs))
            mlflow.log_metric("reranked_docs", len(top_docs))

            mlflow.log_text(question, "question.txt")
            mlflow.log_text(response.content, "answer.txt")

            # ---------------- UI OUTPUT ----------------
            st.subheader("✅ Answer")
            st.write(response.content)

            # 📊 Metrics
            st.subheader("📊 Monitoring")
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱ Latency", latency)
            col2.metric("📄 Retrieved Docs", len(docs))
            col3.metric("🎯 Reranked Docs", len(top_docs))

            # 🔍 Trace
            st.subheader("🔍 Pipeline Trace")
            st.write("Retrieved Docs:", len(docs))
            st.write("After Rerank:", len(top_docs))
            st.write("Context Length:", len(context))

            # 📚 References
            st.subheader("📚 References")
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                st.write(doc.page_content[:300] + "...")

            # 📝 Logs
            log_entry = {
                "question": question,
                "answer": response.content,
                "latency": latency
            }
            st.session_state.logs.append(log_entry)

# -------------------------------------------------
# Log History
# -------------------------------------------------
if st.session_state.logs:
    st.subheader("📜 Query Logs")
    for i, log in enumerate(reversed(st.session_state.logs)):
        st.markdown(f"### Query {i+1}")
        st.write("Question:", log["question"])
        st.write("Answer:", log["answer"])
        st.write("Latency:", log["latency"])
        st.markdown("---")
        