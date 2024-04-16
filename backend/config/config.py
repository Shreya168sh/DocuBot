import os
from dotenv import load_dotenv, find_dotenv


# Load environment variables from .env file (if any)
load_dotenv(find_dotenv())


class PathConfigurations:
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_PATH, "model")
    LOG_DIR = os.path.join(BASE_PATH, "logs")
    DOCUMENTS_PATH = os.path.join(BASE_PATH, "documents")


class PineconeConfigurations:
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
    PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
