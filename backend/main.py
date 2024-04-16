from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import uvicorn
from src.helper import HelperFunctions
from src.document_loader import DocumentHandler
from database import ChatbotDB
from logger import Logger


helper = HelperFunctions()
document = DocumentHandler()
logger = Logger("API")


# Response base model
class Response(BaseModel):
    result: str | None


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# For both document and query.
@app.post("/predict", response_model=Response)
# @app.post("/predict")
def predict(query: str, file: UploadFile = File(None)) -> Any:
    try:
        logger.info(msg="Query received!")
        # load embeddings
        embedding = helper.download_embeddings()

        # Initialise database
        db = ChatbotDB()

        if file:
            logger.info(msg="File received!")
            if file.filename.split(".")[-1] in ["txt", "doc", "pdf", "csv"]:
                file_path = document.save(file=file)

                # Load the document according to its format
                documents = document.load(file_path=file_path)
                
                # Establishing connection and creating index in pinecone
                db.connect()

                # Create chunks from the document
                text_chunks = helper.split_text(documents=documents)

                # Store embeddings of the document in the vector DB
                vectors_stored = db.insert_embeddings(text_chunks=text_chunks, embedding=embedding)

                if not vectors_stored:
                    msg = "INTERNAL SERVER ERROR. Kindly upload the document again!"
                    logger.error(msg=msg)
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
            
            else:
                msg = "Uploded document type not supported!"
                logger.error(msg=msg)
                # raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=msg)
                return {"status_code": status.HTTP_406, "detail": msg}

        # load the LLM model: LLAMA2
        llm = helper.load_model()

        # Fetch the embeddings most similar to query embedding from vector DB
        vector_store = db.get_embeddings(embedding=embedding)

        if not vector_store:
            msg = "INTERNAL SERVER ERROR. Kindly ask the query again!"
            logger.error(msg=msg)
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
            return {"status_code": status.HTTP_500_INTERNAL_SERVER_ERROR, "detail": msg}
        
        prompt = helper.prepare_prompt()

        # Initialise qa chain
        qa = helper.qa_chain(prompt=prompt, 
                    llm=llm,
                    vector_store=vector_store)

        # return search result
        result = helper.search_result(qa=qa, query=query)
        return Response(result=result["result"])
    
    except Exception as e:
        logger.error(msg=f"Error occurred: {str(e)}")
        return {"Error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8082)
        