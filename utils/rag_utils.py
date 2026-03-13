import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from models.embeddings import get_embedding_model
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


def _get_loader(file_path, file_ext):
    ext = file_ext.lower().strip(".")
    if ext == "pdf":
        return PyPDFLoader(file_path)
    elif ext == "txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == "docx":
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type '.{ext}'. Upload a PDF, TXT, or DOCX.")


def load_documents(file_path, file_ext):
    try:
        loader = _get_loader(file_path, file_ext)
        return loader.load()
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading document: {e}")


def split_documents(documents):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        return splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error splitting documents: {e}")


def build_vectorstore(chunks):
    try:
        embedding_model = get_embedding_model()
        return FAISS.from_documents(chunks, embedding_model)
    except Exception as e:
        raise RuntimeError(f"Error building vector store: {e}")


def retrieve_relevant_chunks(query, vectorstore, k=TOP_K):
    # KNN similarity search — returns top-k most relevant chunks
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        raise RuntimeError(f"Error during chunk retrieval: {e}")


def process_uploaded_file(uploaded_file):
    """Loads, chunks and indexes an uploaded file. Returns (vectorstore, chunk_count)."""
    file_ext = uploaded_file.name.rsplit(".", 1)[-1]
    tmp_file = None
    try:
        # Write to temp file so LangChain loaders can open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_file = tmp.name

        documents   = load_documents(tmp_file, file_ext)

        # Replace temp path with the real filename in metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1  # 1-indexed

        chunks      = split_documents(documents)
        vectorstore = build_vectorstore(chunks)

        return vectorstore, len(chunks)

    except Exception as e:
        raise RuntimeError(f"Failed to process '{uploaded_file.name}': {e}")

    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
