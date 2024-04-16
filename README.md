# DocuBot
A document-based chatbot


## Step 01: Clone the repository
git clone https://github.com/Shreya168sh/DocuBot.git


## Step 02: Create and activate a virtual environment
**Using Conda**
conda create -n chatbot python=3.10 -y
conda activate chatbot

OR

**Using venv**
python3 -m venv chatbot
source chatbot/bin/activate (for Ubuntu)


## Step 03: Install the requirements
pip install -r requirements.txt


## Step 04: Create an .env file
**Pinecone DB credentials**
PINECONE_API_KEY = "Enter your API Key here"
PINECONE_API_ENV = "Enter your API Env here"
PINECONE_INDEX = "Enter Index Name here"


## Step 05: Execute the API
a. Go to project directory -
     cd backend
b. Execute main file - 
     python3 main.py


## Step 06: Copy the given URL in your search engine
https://localhost:8082/docs
