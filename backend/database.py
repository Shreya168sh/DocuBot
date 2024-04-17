from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from pinecone.core.client.exceptions import NotFoundException, UnauthorizedException
import time
from config import PineconeConfigurations
from logger import Logger


class ChatbotDB:
    def __init__(
            self,
            api_key=PineconeConfigurations.PINECONE_API_KEY,
            environment=PineconeConfigurations.PINECONE_API_ENV,
            index=PineconeConfigurations.PINECONE_INDEX
            ) -> None:
        """
        Initializes a new instance of the ChatbotDB class.

        Args:
            api_key (str, optional): The API key for Pinecone. Defaults to PineconeConfigurations.PINECONE_API_KEY.
            environment (str, optional): The environment for Pinecone. Defaults to PineconeConfigurations.PINECONE_API_ENV.
            index (str, optional): The index name for Pinecone. Defaults to PineconeConfigurations.PINECONE_INDEX.

        Returns:
            None
        """
        
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
        """
        Establishes a connection with Pinecone, checks for an existing index, deletes the index if it exists, 
        creates a new index if needed, initializes the index, and handles exceptions for connection and deletion.

        Returns:
            None
        """
        if self.use_serverless:  
            spec = ServerlessSpec(cloud='aws', region='us-west-2')  
        else:  
            # if not using a starter index, you should specify a pod_type too  
            spec = PodSpec(environment=self.environment)  

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
                    self.logger.info(msg="Deleted existing data in the index.")

                except NotFoundException:
                    self.logger.error(msg="Pinecone index not found, skipping deletion.")

        else:
            self.logger.info(msg="Creating new index...")
            # create a new index
            self.client.create_index(
                name=self.index,
                dimension=384,  # dimensionality of all-MiniLM-L6-v2
                spec=spec
            )

        self.logger.info(msg="Initialising index...")
        # wait for index to be initialized
        while not self.client.describe_index(self.index).status['ready']:
            time.sleep(1)

        self.logger.info(msg="Pinecone index is ready to store embeddings.")

        try:      
            index = self.client.Index(self.index)
            self.logger.info(msg="Connection established successfully!")

        except UnauthorizedException:
            self.logger.error(msg="Unauthorized connection, kindly provide valid API KEY and API ENV.")
    
    # Store embeddings of the document in the vector DB
    def insert_embeddings(self, text_chunks, embedding):
        """
        Inserts embeddings of text chunks into the Pinecone vector store.

        Parameters:
            text_chunks (List[TextChunk]): A list of TextChunk objects representing the text chunks to be inserted.
            embedding (Embedding): The embedding to be used for the text chunks.

        Returns:
            bool: True if the embeddings were successfully inserted, False otherwise.
        """
        try:
            PineconeVectorStore.from_texts(texts=[t.page_content for t in text_chunks], index_name=self.index, embedding=embedding)
            return True

        except Exception as e:
            self.logger.error(msg=f"Error while inserting embeddings: {str(e)}")
            return False

    # Fetch embeddings of the most similar texts with query from vector DB
    def get_embeddings(self, embedding):
        """
        Retrieves the embeddings from the Pinecone vector store based on the given embedding.

        Parameters:
            embedding (Embedding): The embedding to be used for retrieving the embeddings.

        Returns:
            PineconeVectorStore or False: The PineconeVectorStore object containing the embeddings if successful,
                                         False otherwise.

        Raises:
            Exception: If there is an error while fetching the embeddings.
        """
        try:
            vector_store = PineconeVectorStore.from_existing_index(index_name=self.index, embedding=embedding)
            return vector_store

        except Exception as e:
            self.logger.error(msg=f"Error while fetching embeddings: {str(e)}")
            return False
