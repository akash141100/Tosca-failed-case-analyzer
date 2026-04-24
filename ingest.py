import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the Data
print("Loading Tosca logs...")
# We use DirectoryLoader to grab every .txt file inside our test_logs folder
loader = DirectoryLoader('./test_logs', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} log files.")

# 2. Chop the Data into Chunks
print("Splitting text into chunks...")
# We split the text so the AI doesn't get overwhelmed. 
# Overlap ensures we don't accidentally cut a stack trace in half.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# 3. Initialize the Translator (Embeddings)
print("Initializing embedding model...")
# This converts our English text into math (vectors)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Save to ChromaDB
print("Building Chroma Database...")
# This creates a local folder called 'chroma_db' and saves all our vectors there
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("✅ Ingestion Complete! Your database is ready.")