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


logger = Logger("HelperFunctions")

class HelperFunctions:
    def __init__(
            self,
            ) -> None:
        self.logger = Logger("HelperFunctions")
        self.base_path = PathConfigurations.BASE_PATH,
        self.model_path = PathConfigurations.MODEL_PATH

    # Create text chunks
    def split_text(self, documents):
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
        try:
            self.logger.info("Downloading Embeddings from HuggingfaceHub...")
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.logger.info("Embeddings Downloaded!")
            return embedding
        
        except Exception as e:
            self.logger.error(msg=f"Error while downloading embeddings: {str(e)}")

    # Download llm model from Huggingface Hub
    def download_model(self):
        try:
            # print("Downloading model...")
            self.logger.info("Downloading LLAMA2...")
            # print(f"model path in download model: {self.model_path}")
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
        try:
            print(f"model path: {self.model_path}")

            if (not os.path.exists(self.model_path)) or \
                (os.path.exists(self.model_path) and \
                not any(fname.endswith('.bin') for fname in os.listdir(self.model_path))):
                self.download_model()

            # print("Loading model...")
            self.logger.info("Loading LLAMA2...")
            model = os.path.join(self.model_path, os.listdir(self.model_path)[0])
            print(f"model: {model}")
            llm = CTransformers(model=model,
                                model_type="llama",
                                config={"max_new_tokens": 512,
                                        "temperature": 0})

            # print("Model loaded successfully!")
            self.logger.info("LLAMA2 Loaded!")
            return llm
        
        except Exception as e:
            self.logger.error(msg=f"Error while loading model: {str(e)}")


    def prepare_prompt(self):
        try:
            # Create prompt template
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            self.logger.info(msg="Prompt created!")
            return prompt
        except Exception as e:
            self.logger.error(msg=f"Error while preparing prompt: {str(e)}")


    # create qa chain for question answering chatbot
    def qa_chain(self, prompt, llm, vector_store):
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
        try:
            self.logger.info(msg="Searching answer...")
            response = qa({"query": query})
            self.logger.info(msg="Query resolved!")
            return response
        except Exception as e:
            self.logger.error(msg=f"Error while resolving query: {str(e)}")


# if __name__ == "__main__":
#     print(HelperFunctions().download_embeddings())
#     print(HelperFunctions().load_model())