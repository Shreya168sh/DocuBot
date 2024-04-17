from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from logger.logger import Logger
import os
from src.prompt import prompt_template
from config import PathConfigurations


class HelperFunctions:
    def __init__(
            self,
            ) -> None:
        """
        Initializes the HelperFunctions object with logger, base path, and model path.
        """
        self.logger = Logger("HelperFunctions")
        self.base_path = PathConfigurations.BASE_PATH,
        self.model_path = PathConfigurations.MODEL_PATH

    # Create text chunks
    def split_text(self, documents):
        """
        Splits the given documents into text chunks using the RecursiveCharacterTextSplitter.

        Parameters:
            documents (list): A list of documents to be split into text chunks.

        Returns:
            list: A list of text chunks.

        Raises:
            Exception: If there is an error while tokenizing the documents.
        """
        try:
            self.logger.info("Tokenization in progress...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            docs = text_splitter.split_documents(documents=documents)
            self.logger.info("Tokenization completed!")
            return docs

        except Exception as e:
            self.logger.error(msg=f"Error while tokenizing: {str(e)}")


    # Download embedding model from Huggingface Hub
    def download_embeddings(self):
        """
        Downloads embeddings from HuggingfaceHub and returns the embedding model.

        Returns:
            HuggingFaceEmbeddings: The embedding model downloaded from HuggingfaceHub.

        Raises:
            Exception: If there is an error while downloading the embeddings.
        """
        try:
            self.logger.info("Downloading Embeddings from HuggingfaceHub...")
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.logger.info("Embeddings Downloaded!")
            return embedding
        
        except Exception as e:
            self.logger.error(msg=f"Error while downloading embeddings: {str(e)}")

    # Download llm model from Huggingface Hub
    def download_model(self):
        """
        Downloads the LLAMA2 model from Huggingface Hub.

        Returns:
            str: The path to the downloaded model.

        Raises:
            Exception: If there is an error while downloading the model.
        """
        try:
            self.logger.info("Downloading LLAMA2...")
            os.makedirs(self.model_path, exist_ok=True)
            
            model_path = hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-Chat-GGML",
                filename="llama-2-7b-chat.ggmlv3.q2_K.bin",
                local_dir=self.model_path
                )

            return model_path

        except Exception as e:
            self.logger.error(msg=f"Error while downloading model: {str(e)}")
            

    # Load the model
    def load_model(self):
        """
        Load the LLAMA2 model.

        This function checks if the model has already been downloaded and if not, it downloads it.
        It then loads the model and returns the loaded model object.

        Returns:
            CTransformers: The loaded LLAMA2 model.

        Raises:
            Exception: If there is an error while loading the model.
        """
        try:
            if (not os.path.exists(self.model_path)) or \
                (os.path.exists(self.model_path) and \
                not any(fname.endswith('.bin') for fname in os.listdir(self.model_path))):
                self.download_model()

            self.logger.info("Loading LLAMA2...")
            model = os.path.join(self.model_path, os.listdir(self.model_path)[0])
            llm = CTransformers(model=model,
                                model_type="llama",
                                config={"max_new_tokens": 512,
                                        "temperature": 0})

            self.logger.info("LLAMA2 Loaded!")
            return llm
        
        except Exception as e:
            self.logger.error(msg=f"Error while loading model: {str(e)}")


    def prepare_prompt(self):
        """
        Prepares the prompt for the chatbot.

        Returns:
            PromptTemplate: The prepared prompt template.

        Raises:
            Exception: If there is an error while preparing the prompt.
        """
        try:
            # Create prompt template
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            self.logger.info(msg="Prompt created!")
            return prompt

        except Exception as e:
            self.logger.error(msg=f"Error while preparing prompt: {str(e)}")


    # create qa chain for question answering chatbot
    def qa_chain(self, prompt, llm, vector_store):
        """
        Initializes a RetrievalQA chain for question answering.

        Args:
            prompt (str): The prompt to set in the chain type.
            llm (object): The language model to use for question answering.
            vector_store (object): The vector store to use for retrieval.

        Returns:
            RetrievalQA: The initialized RetrievalQA chain.

        Raises:
            Exception: If there is an error while creating the QA chain.
        """
        try:
            # Set prompt into chain type
            chain_type_kwargs = {"prompt": prompt}
            
            # Initialise RetrievalQA for question answering
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={'k': 1}),
                return_source_documents=False,
                chain_type_kwargs=chain_type_kwargs
            )
            self.logger.info(msg="QA chain created!")
            return qa

        except Exception as e:
            self.logger.error(msg=f"Error while creating QA chain: {str(e)}")


    # return most appropriate answer of the query from stored vectors if found
    def search_result(self, qa, query):
        """
        Searches for an answer to a given query using a question-answering model.

        Args:
            qa (object): The question-answering model to use for searching the answer.
            query (str): The query to search for an answer to.

        Returns:
            dict: The response containing the answer to the query.

        Raises:
            Exception: If there is an error while resolving the query.
        """
        try:
            self.logger.info(msg="Searching answer...")
            response = qa({"query": query})
            self.logger.info(msg="Query resolved!")
            return response

        except Exception as e:
            self.logger.error(msg=f"Error while resolving query: {str(e)}")
