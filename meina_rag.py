import os
import glob # For finding files
import ollama # Official Ollama Python client
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

# --- Configuration ---
DATA_PATH = "data"
DB_PATH = "db"
LLM_MODEL_NAME = "qwen3:8b" 
EMBEDDING_MODEL_NAME = "nomic-embed-text" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PROCESSED_FILES_LOG = os.path.join(DB_PATH, "processed_files.log")
COLLECTION_NAME = "ollama_rag_store" # Chroma collection name

# --- RAG Prompt Template ---
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

Answer the question based ONLY on the context provided. If the context doesn't contain the answer, state that you cannot answer based on the provided context.
"""

def check_ollama_availability(model_name):
    """Checks if the specified Ollama model is available."""
    try:
        ollama.show(model_name)
        print(f"Model '{model_name}' is available.")
        return True
    except Exception as e: # Catch broader exception for Ollama API errors
        if hasattr(e, 'status_code') and e.status_code == 404: # More specific check for ollama client
             print(f"Error: Model '{model_name}' not found. Please pull it using 'ollama pull {model_name}'")
        elif "model not found" in str(e).lower(): # Fallback for other error structures
            print(f"Error: Model '{model_name}' not found. Please pull it using 'ollama pull {model_name}'")
        else:
            print(f"Error interacting with Ollama for model '{model_name}': {e}")
            print("Please ensure Ollama service is running and accessible.")
        return False

def get_all_data_files(data_path):
    """Gets all .txt and .pdf file paths from the data directory, sorted for consistency."""
    all_files = []
    if not os.path.exists(data_path):
        print(f"Warning: Data directory '{data_path}' does not exist. Creating it.")
        try:
            os.makedirs(data_path)
            print(f"Successfully created data directory: {data_path}")
        except OSError as e:
            print(f"Error: Could not create data directory {data_path}: {e}")
            return [] # Return empty list if directory cannot be created
    
    # Scan for .txt files
    for ext_pattern in ("**/*.txt", "**/*.TXT"): # Case-insensitive for extensions
        all_files.extend(glob.glob(os.path.join(data_path, ext_pattern), recursive=True))
    
    # Scan for .pdf files
    for ext_pattern in ("**/*.pdf", "**/*.PDF"): # Case-insensitive for extensions
        all_files.extend(glob.glob(os.path.join(data_path, ext_pattern), recursive=True))
        
    # Return absolute paths to ensure uniqueness and correct tracking
    return sorted([os.path.abspath(f) for f in all_files])


def load_processed_files(log_path):
    """Loads the set of processed file paths from the log file."""
    if not os.path.exists(log_path):
        return set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except IOError as e:
        print(f"Error reading processed files log {log_path}: {e}. Assuming no files processed.")
        return set()

def update_processed_files_log(file_paths, log_path):
    """Adds newly processed file paths to the log file."""
    try:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for path in file_paths:
                f.write(path + "\n")
        print(f"Updated processed files log: {log_path}")
    except IOError as e:
        print(f"Error updating processed files log {log_path}: {e}")


def load_specific_documents(file_paths):
    """Loads specific documents from a list of file paths."""
    all_documents = []
    print(f"Loading {len(file_paths)} specific document(s)...")
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found, skipping: {file_path}")
                continue
            if file_path.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                all_documents.extend(loader.load())
            elif file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path) # PyPDFLoader handles its own errors for corrupt files often
                docs_from_pdf = loader.load()
                if not docs_from_pdf: # PyPDFLoader might return empty list for unreadable PDFs
                    print(f"Warning: Could not extract text from PDF (or PDF is empty): {file_path}")
                all_documents.extend(docs_from_pdf)
            else:
                print(f"Warning: Unsupported file type skipped: {file_path}")
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
    print(f"Successfully loaded {len(all_documents)} new document(s) from {len(file_paths)} specified file(s).")
    return all_documents

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits documents into smaller chunks."""
    if not documents:
        return []
    print(f"Splitting {len(documents)} document(s) into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    return docs

def get_embeddings_model(model_name=EMBEDDING_MODEL_NAME):
    """Initializes the Ollama embeddings model."""
    print(f"Initializing embeddings model: {model_name}")
    return OllamaEmbeddings(model=model_name)

def initialize_vector_store(db_path, embeddings_model, collection_name):
    """Initializes or loads a Chroma vector store."""
    if not os.path.exists(db_path):
        print(f"Database directory {db_path} not found, creating it.")
        try:
            os.makedirs(db_path)
        except OSError as e:
            print(f"Error: Could not create database directory {db_path}: {e}. Exiting.")
            return None # Critical error, cannot proceed

    print(f"Initializing/loading vector store from: {db_path} with collection: {collection_name}")
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=embeddings_model
        )
        print("Vector store initialized/loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error initializing/loading vector store: {e}")
        return None


def add_chunks_to_vector_store(vector_store, doc_chunks):
    """Adds new document chunks to the vector store."""
    if not doc_chunks:
        print("No document chunks to add to vector store.")
        return False
    if vector_store is None:
        print("Error: Vector store is not initialized. Cannot add documents.")
        return False

    print(f"Adding {len(doc_chunks)} chunks to the vector store...")
    try:
        vector_store.add_documents(documents=doc_chunks)
        vector_store.persist() # Persist changes
        print("Chunks added and vector store persisted.")
        return True
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return False

def get_retriever(vector_store):
    """Gets the retriever from the vector store."""
    if vector_store is None:
        print("Error: Vector store not initialized. Cannot create retriever.")
        return None
    print("Creating retriever...")
    return vector_store.as_retriever()

def setup_rag_chain(retriever, llm_model_name=LLM_MODEL_NAME, rag_prompt_template=RAG_PROMPT_TEMPLATE):
    """Sets up the RAG chain using Langchain Expression Language (LCEL)."""
    if retriever is None:
        print("Error: Retriever not available. Cannot set up RAG chain.")
        return None
    print(f"Setting up RAG chain with LLM: {llm_model_name}")
    try:
        llm = ChatOllama(model=llm_model_name)
        prompt = ChatPromptTemplate.from_template(rag_prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain setup complete.")
        return rag_chain
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return None

def main():
    print("Starting RAG application setup...")

    # --- Create DATA_PATH and DB_PATH if they don't exist ---
    # DATA_PATH is handled by get_all_data_files
    # DB_PATH is handled by initialize_vector_store and update_processed_files_log

    # --- Check Model Availability ---
    if not check_ollama_availability(LLM_MODEL_NAME):
        return 
    if not check_ollama_availability(EMBEDDING_MODEL_NAME):
        return

    # --- Initialize Embeddings Model ---
    embeddings_model = get_embeddings_model()

    # --- Initialize/Load Vector Store ---
    vector_store = initialize_vector_store(DB_PATH, embeddings_model, COLLECTION_NAME)
    if vector_store is None:
        print("Failed to initialize vector store. Exiting.")
        return

    # --- Identify and Process New Files ---
    print(f"Checking for new files in: {DATA_PATH}")
    processed_file_paths = load_processed_files(PROCESSED_FILES_LOG)
    print(f"Found {len(processed_file_paths)} previously processed files in log.")
    
    all_current_files_in_data = get_all_data_files(DATA_PATH)
    print(f"Found {len(all_current_files_in_data)} total files in data directory.")

    new_files_to_process = sorted(list(set(all_current_files_in_data) - processed_file_paths))

    if new_files_to_process:
        print(f"Found {len(new_files_to_process)} new file(s) to process:")
        for f_path in new_files_to_process:
            print(f"  - {os.path.basename(f_path)}") # Print just filename for brevity
        
        documents_from_new_files = load_specific_documents(new_files_to_process)

        if documents_from_new_files:
            new_doc_chunks = split_documents(documents_from_new_files)
            if new_doc_chunks: # Ensure there are chunks to add
                if add_chunks_to_vector_store(vector_store, new_doc_chunks):
                    update_processed_files_log(new_files_to_process, PROCESSED_FILES_LOG)
                    print("Successfully processed and added new files to the vector store.")
            else:
                print("No splittable content found in new files. No update to vector store.")
        else:
            print("No content loaded from new files. Vector store not updated with these files.")
    else:
        print("No new files found in data directory. Vector store is up-to-date with tracked files.")

    # --- RAG Chain Setup ---
    # Check if vector store is empty before setting up RAG
    try:
        collection_count = vector_store._collection.count()
        if collection_count == 0:
            print("\nWarning: Vector store is empty.")
            print(f"Please add .txt or .pdf documents to the '{DATA_PATH}' directory and restart the application.")
            print("The RAG system will not be able to answer questions without any data.")
    except Exception as e:
        print(f"Could not get collection count from vector store: {e}. Proceeding...")


    retriever = get_retriever(vector_store)
    rag_chain = setup_rag_chain(retriever)

    if rag_chain is None:
        print("Failed to set up RAG chain. Exiting.")
        return

    # --- Interactive Query Loop ---
    print("\n--- RAG Query Interface ---")
    print(f"Using LLM: {LLM_MODEL_NAME} | Embeddings: {EMBEDDING_MODEL_NAME}")
    print("Enter 'quit' to exit.")
    
    while True:
        query = input("Ask a question about your documents: ")
        if query.lower() == 'quit':
            break
        if not query.strip():
            continue

        print("\nThinking...")
        try:
            if vector_store._collection.count() == 0:
                print("\nAnswer:")
                print("The document database is currently empty. I cannot answer questions until documents are added.")
                print(f"Please add .txt or .pdf files to the '{DATA_PATH}' directory and restart the application.")
            else:
                response = rag_chain.invoke(query)
                print("\nAnswer:")
                print(response)
        except Exception as e:
            print(f"\nError during RAG chain invocation: {e}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
