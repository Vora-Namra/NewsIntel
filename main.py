import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("📰 News Research App")
st.sidebar.title("News Article URLs")

# URL input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

main_placeholder = st.empty()

# Process URLs
if st.sidebar.button("Process URLs"):
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔄 Loading and parsing URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        main_placeholder.text("✂️ Splitting documents...")
        docs = text_splitter.split_documents(data)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("📦 Saving vector store...")
        vectorstore.save_local("faiss_index")

        main_placeholder.success("✅ Processing complete!")

# Question input
query = st.text_input("Ask a question based on the articles:")

if query:
    if os.path.exists("faiss_index/index.faiss"):  # Check index file
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        retriever = vectorstore.as_retriever()

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        st.write("🤖 Getting answer from Gemini...")
        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("🧠 Answer:")
        st.write(result["answer"])

        st.subheader("📚 Sources:")
        if "sources" in result and result["sources"]:
            for source in result["sources"].split("\n"):
                st.write(source)
        else:
            st.write("No sources found.")
    else:
        st.error("❌ No FAISS index found. Please process the URLs first.")
