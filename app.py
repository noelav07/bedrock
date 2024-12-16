import json
import os
import boto3
import streamlit as st

# We will use Titan Embeddings Model v2 to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion function that processes both PDF and TXT files from the 'data' directory
def data_ingestion():
    documents = []

    # Load PDF documents if they exist
    if any(filename.endswith(".pdf") for filename in os.listdir("data")):
        pdf_loader = PyPDFDirectoryLoader("data")
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)

    # Load text files if they exist
    if any(filename.endswith(".txt") for filename in os.listdir("data")):
        text_documents = []
        for filename in os.listdir("data"):
            if filename.endswith(".txt"):
                with open(os.path.join("data", filename), "r") as file:
                    content = file.read()
                    text_documents.append(content)
                    print(f"Loaded text file: {filename} with content: {content[:100]}...")  # Log content
        documents.extend(text_documents)

    # If no documents were found, return an empty list
    if not documents:
        return []

    # Character split works better with this dataset
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    print(f"Total documents after splitting: {len(docs)}")  # Log number of documents after splitting
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    print("Generating embeddings for the documents...")
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    print("Saving vector store...")
    vectorstore_faiss.save_local("faiss_index")
    print("Vector store saved.")

# Llama3 Model (for response generation)
def get_llama3_llm():
    llm = Bedrock(model_id="us.meta.llama3-2-90b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 20 words with detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Get response from the LLM based on the local vector store and the query
def get_response_llm(llm, vectorstore_faiss, query):
    # Retrieve relevant context from the vector store using the query
    docs = vectorstore_faiss.similarity_search(query, k=3)  # Adjust 'k' for the number of documents to fetch
    context = "\n".join([doc.page_content for doc in docs])

    # Prepare the prompt using the retrieved context
    prompt = PROMPT.format(context=context, question=query)

    # Call the LLM with the prompt
    payload = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)

    response = bedrock.invoke_model(
        body=body,
        modelId=llm.model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    response_text = response_body.get('generation') or response_body.get('outputs') or response_body.get('completions', 'No generation returned')
    
    return response_text.strip()  # Return the response without additional formatting

# Streamlit UI
def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()  # Ingest data from the local files (PDF and TXT)
                get_vector_store(docs)  # Update the vector store with the local documents
                st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            # Load the local vector store and use it for querying
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))  # Query using local vector store
            st.success("Done")

if __name__ == "__main__":
    main()
