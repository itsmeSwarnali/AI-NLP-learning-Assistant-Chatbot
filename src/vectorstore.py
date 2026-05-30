
# Documents → chunks → embeddings → Chroma vector database


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vectorstore(documents, persist_directory = "vectorstore/chroma_db"):
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 100
    )
    ## my ingestion function already returns LangChain Document objects. 
    ## create_documents expects raw strings, 
    ## while split_documents splits existing Document objects and preserves metadata.
    chunks = splitter.split_documents(documents)

    # Create Enbeddings

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )

    return vectorstore