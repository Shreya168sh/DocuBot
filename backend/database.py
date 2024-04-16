from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from pinecone.core.client.exceptions import NotFoundException, UnauthorizedException
import time
from config import PineconeConfigurations
from logger import Logger


# print(f"key: {PINECONE_API_KEY}")
# print(f"env: {PINECONE_API_ENV}")


class ChatbotDB:
    def __init__(
            self,
            # logger: BoundLogger,
            api_key=PineconeConfigurations.PINECONE_API_KEY,
            environment=PineconeConfigurations.PINECONE_API_ENV,
            index=PineconeConfigurations.PINECONE_INDEX
            ) -> None:
        
        # configure client
        self.client = Pinecone(
            api_key=api_key,
            environment=environment
            )
        self.use_serverless = False
        self.api_key = api_key
        self.environment = environment
        self.index = index
        self.logger = Logger(name="PineconeDB")
    
    # Initialize the Pinecone
    def connect(self):
        if self.use_serverless:  
            spec = ServerlessSpec(cloud='aws', region='us-west-2')  
        else:  
            # if not using a starter index, you should specify a pod_type too  
            spec = PodSpec(environment=self.environment)  
        
        # print("Establishing connection with Pinecone...")
        self.logger.info(msg="Establishing connection with Pinecone...")

        # check for and delete index if already exists
        if self.index in self.client.list_indexes().names():
            # Delete existing data
            index = self.client.Index(self.index)
            vector_status = index.describe_index_stats()
            vector_count = vector_status["total_vector_count"]
            if vector_count > 0:
                try:
                    index.delete(deleteAll=True)
                    # print("\nDeleted existing data in the index.")
                    self.logger.info(msg="Deleted existing data in the index.")
                except NotFoundException:
                    # print("Pinecone index not found. Skipping deletion.")
                    self.logger.error(msg="Pinecone index not found, skipping deletion.")

        else:
            # print("Creating new index...")
            self.logger.info(msg="Creating new index...")
            # create a new index
            self.client.create_index(
                name=self.index,
                dimension=384,  # dimensionality of all-MiniLM-L6-v2
                spec=spec
            )

        # print(f"Initialising index...")
        self.logger.info(msg="Initialising index...")
        # wait for index to be initialized
        while not self.client.describe_index(self.index).status['ready']:
            # print(f"index status['ready']?: {self.client.describe_index(self.index).status['ready']}")
            time.sleep(1)

        self.logger.info(msg="Pinecone index is ready to store embeddings.")

        try:      
            index = self.client.Index(self.index)
            # print("Connection established successfully!")
            self.logger.info(msg="Connection established successfully!")
            # print(f"Pinecone Index Details:\n{index.describe_index_stats()}")
        except UnauthorizedException:
            # print("Unauthorized connection, kindly provide valid API KEY and API ENV")
            self.logger.error(msg="Unauthorized connection, kindly provide valid API KEY and API ENV.")
    
    # Store embeddings of the document in the vector DB
    def insert_embeddings(self, text_chunks, embedding):
        try:
            PineconeVectorStore.from_texts(texts=[t.page_content for t in text_chunks], index_name=self.index, embedding=embedding)
            return True
        except Exception as e:
            self.logger.error(msg=f"Error while inserting embeddings: {str(e)}")
            return False

    # Fetch embeddings of the most similar texts with query from vector DB
    def get_embeddings(self, embedding):
        try:
            vector_store = PineconeVectorStore.from_existing_index(index_name=self.index, embedding=embedding)
            return vector_store

        except Exception as e:
            self.logger.error(msg=f"Error while fetching embeddings: {str(e)}")
            return False
    

# if __name__ == "__main__":
#     obj = ChatbotDB()
#     print(obj.connect())