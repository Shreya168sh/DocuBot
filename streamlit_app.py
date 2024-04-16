import streamlit as st
import time
import random
import os
from src.helper import *
from src.document_loader import save_document, load_document
from database import ChatbotDB


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    vectors_stored = False
    response = "False"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")

    with col2:
        # st.title("🤖DocuBot")
        st.title("🕵Docubot")

    with col3:
        st.write("")
        
    st.write("\n")

    # load embeddings
    embedding = download_embeddings()

    # Initialise database
    db = ChatbotDB()

    # Request file from user
    uploaded_file = st.file_uploader("Choose a file", type=("txt", "doc", "pdf", "csv"))

    # Initialize chat history
    if "file" not in st.session_state:
        st.session_state.file = True

    if uploaded_file and st.session_state.file:
        with st.spinner('Preparing document...'):
            while not vectors_stored:
                file_path = save_document(base_path=base_path, file=uploaded_file)
                print(f"file path: {file_path}")

                # Load the document according to its format
                documents = load_document(file_path=file_path)
                print(f"\ndocs: {len(documents)}")

                # Establishing connection and creating index in pinecone
                db.connect()

                # Create chunks from the document
                text_chunks = split_text(documents=documents)
                print(f"\ntext chunks: {len(text_chunks)}")

                # Store embeddings of the document in the vector DB
                vectors_stored = db.insert_embeddings(text_chunks=text_chunks, embedding=embedding)
                st.session_state.file = False if vectors_stored else True
                
    # query from user
    query = st.text_input(
        "Ask your question",
        placeholder="Can you give me a brief summary?",
        disabled=not uploaded_file)

    # Generate response for the user's query
    if query:
        with st.spinner('Generating response...'):
            while response == "False":
                # load the LLM model: LLAMA2
                llm = load_model(base_path=base_path)

                # Fetch the embeddings most similar to query embedding from vector DB
                vector_store = db.get_embeddings(embedding=embedding)

                # Prepare the user's prompt
                prompt = prepare_prompt()

                # Initialise qa chain
                qa = qa_chain(prompt=prompt, 
                                llm=llm,
                                vector_store=vector_store)

                # return search result
                response = search_result(qa=qa, query=query)
                st.write(response["result"])


# # Streamed response emulator
# def response_generator():
#     response = random.choice(
#         [
#             "Hello there, how can I assist you today?",
#             "Hi Buddy! Is there anything I can help you with?",
#             "Do you need any help?"
#         ]
#     )
    
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)


    # with st.chat_message("assistant"):
    #     st.write("Hi, how can I help you?")

    # # Initialize chat history
    # if "messages" not in st.session_state:
    #     st.session_state = []

    # else:
    #     # Display chat messages from history on app run
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):        # role -> author of the message
    #             st.markdown(message["content"])           # content -> content of the message



    # # React to user input
    # if prompt := st.chat_input("Say something"):        # := -> will check if prompt variable is none
    #     # Display user message in chat message container
    #     st.chat_message("user").markdown(prompt)

    #     # Add user's message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})
        
    #     # Initialize chat history
    #     if "messages" not in st.session_state:
    #         st.session_state = []

    #     else:
    #         # Display assistant's reponse in chat message container
    #         with st.chat_message("assistant"):
    #             response = st.write(response_generator())

    #         # Add assistant's reponse to chat history
    #         st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        st.write(f"Error occurred: {str(e)}")
