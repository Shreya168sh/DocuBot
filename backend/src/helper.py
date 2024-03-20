from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# To load text documents
def text_loader(file_path):
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    return documents


# To load word documents
def doc_loader(file_path):
    loader = Docx2txtLoader(file_path=file_path)
    documents = loader.load()
    return documents


# To load pdf documents
def pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    return documents


# To load csv documents
def csv_loader(file_path):
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        })
    documents = loader.load()
    return documents


# Create text chunks
def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download embedding model from Huggingface Hub
def download_embeddings():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding
