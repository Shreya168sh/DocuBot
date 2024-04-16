from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import  CSVLoader
import os
import shutil
from datetime import datetime
from logger.logger import Logger
from config.config import PathConfigurations


class DocumentHandler:
    def __init__(
            self
            ) -> None:
        self.logger = Logger("DocumentLoader")
        self.base_path = PathConfigurations.BASE_PATH,
        self.model_path = PathConfigurations.MODEL_PATH
        self.document_path = PathConfigurations.DOCUMENTS_PATH

    # To load text documents
    def text_loader(self, file_path):
        loader = TextLoader(file_path=file_path)
        documents = loader.load()
        self.logger.info(msg="Text file loaded!")
        return documents

    # To load word documents
    def doc_loader(self, file_path):
        loader = Docx2txtLoader(file_path=file_path)
        documents = loader.load()
        self.logger.info(msg="Word file loaded!")
        return documents

    # To load pdf documents
    def pdf_loader(self, file_path):
        loader = PyMuPDFLoader(file_path=file_path)
        documents = loader.load()
        self.logger.info(msg="PDF file loaded!")
        return documents

    # To load csv documents
    def csv_loader(self, file_path):
        loader = CSVLoader(
            file_path=file_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
            })
        documents = loader.load()
        self.logger.info(msg="CSV file loaded!")
        return documents

    # To save documents
    def save(self, file):
        try:
            os.makedirs(self.document_path, exist_ok=True)
            
            # for FastAPI
            filename, ext = file.filename.split(".")

            # # for Streamlit
            # filename, ext = file.name.split(".")

            filename = filename + "-" + datetime.strftime(datetime.now(), format="%d-%m-%Y-%H-%M-%S") + "." + ext
            file_path = os.path.join(self.document_path, filename)

            # for FASTAPI
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # # for Streamlit
            # # To read file as bytes:
            # bytes_data = file.getvalue()
            # with open(file_path, "wb") as buffer:
            #     buffer.write(bytes_data)

            self.logger.info(msg="Document saved!")
            return file_path
        
        except Exception as e:
            self.logger.error(msg=f"Error while saving document: {str(e)}")


    # To load documents
    def load(self, file_path):
        try:
            print(f"file path in load docs: {file_path}")
            ext = file_path.split(".")[-1]
            print(f"file path in load docs: {file_path}")
            if ext == "txt":
                documents = self.text_loader(file_path=file_path)
            elif ext == "doc":
                documents = self.doc_loader(file_path=file_path)
            elif ext == "pdf":
                documents = self.pdf_loader(file_path=file_path)
            elif ext == "csv":
                documents = self.csv_loader(file_path=file_path)
            else:
                documents = "File type not supported!"
                self.logger.info(msg="Uploaded document type not supported!")

            return documents
        
        except Exception as e:
            self.logger.error(msg=f"Error while loading document: {str(e)}")
