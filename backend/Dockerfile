FROM python:3.10.8-slim-buster
WORKDIR /backend
COPY . .
RUN apt-get update
RUN pip install -r requirements.txt
# RUN pip install ctransformers --no-binary ctransformers --no-cache-dir
RUN apt-get install libc-bin libc6
RUN apt-get install nano
ENV PINECONE_API_KEY="3fe11a56-ed9c-4529-99f7-ffc77781054c"
ENV PINECONE_API_ENV="gcp-starter"
ENV PINECONE_INDEX="docubot"
EXPOSE 8082
CMD ["streamlit", "run", "main.py"]
