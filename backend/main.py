import streamlit as st
from src.helper import HelperFunctions
from src.document_loader import DocumentHandler
from database import ChatbotDB
from logger import Logger


helper = HelperFunctions()
document = DocumentHandler()
logger = Logger("API")


def main():
    """
    A function to execute the main logic of the program, which involves loading embeddings, 
    initializing the database, handling file uploads, querying users, 
    and generating responses based on the queries.

    Parameters:
    None
    """
    vectors_stored = False
    response = "False"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")

    with col2:
        st.title("🕵Docubot")

    with col3:
        st.write("")
        
    st.write("\n")

    # load embeddings
    embedding = helper.download_embeddings()

    # Initialise database
    db = ChatbotDB()

    # Request file from user
    uploaded_file = st.file_uploader("Choose a file", type=("txt", "doc", "pdf", "csv"))

    # Initialize chat history
    if "file" not in st.session_state:
        st.session_state.file = True

    if uploaded_file and st.session_state.file:
        logger.info(msg="File received!")
        with st.spinner('Preparing document...'):
            while not vectors_stored:
                file_path = document.save(file=uploaded_file)

                # Load the document according to its format
                documents = document.load(file_path=file_path)

                # Establishing connection and creating index in pinecone
                db.connect()

                # Create chunks from the document
                text_chunks = helper.split_text(documents=documents)

                # Store embeddings of the document in the vector DB
                vectors_stored = db.insert_embeddings(text_chunks=text_chunks, embedding=embedding)
                st.session_state.file = False if vectors_stored else True

                if not vectors_stored:
                    msg = "INTERNAL SERVER ERROR. Kindly upload the document again!"
                    logger.error(msg=msg)
                    st.error(msg)
                break
                
    # query from user
    query = st.text_input(
        "Ask your question",
        placeholder="Can you give me a brief summary?",
        disabled=not uploaded_file)

    # Generate response for the user's query
    if query:
        logger.info(msg="Query received!")
        with st.spinner('Generating response...'):
            while response == "False":
                # load the LLM model: LLAMA2
                llm = helper.load_model()

                # Fetch the embeddings most similar to query embedding from vector DB
                vector_store = db.get_embeddings(embedding=embedding)

                if not vector_store:
                    msg = "INTERNAL SERVER ERROR. Kindly ask the query again!"
                    logger.error(msg=msg)
                    st.error(msg)

                # Prepare the user's prompt
                prompt = helper.prepare_prompt()

                # Initialise qa chain
                qa = helper.qa_chain(prompt=prompt, 
                    llm=llm,
                    vector_store=vector_store)

                # return search result
                result = helper.search_result(qa=qa, query=query)
                st.write(result["result"])
                break


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
