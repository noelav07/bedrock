import json
import os
import sys
import boto3
import streamlit as st

# We will be using Titan Embeddings Model To generate Embedding
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

# LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Define your LLM models
def get_llama2_llm():
    # create the Llama2 Model
    llm = Bedrock(model_id="us.meta.llama3-2-90b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    return llm

# Prompt template for querying the model
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    # Retrieve context from the vector store based on the query
    retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    context = retriever.get_relevant_documents(query)
    context_text = " ".join([doc.page_content for doc in context])  # Access the text content

    # Create the prompt from the template
    prompt = PROMPT.format(context=context_text, question=query)

    # Prepare the input payload for Bedrock, with the required 'prompt' key
    input_payload = {
        "prompt": prompt,  # Correct the key to 'prompt'
        "parameters": {
            "max_new_tokens": 512,
            "top_p": 0.9,
            "temperature": 0.6
        }
    }

    # Invoke the model and get the response
    try:
        response = llm._call(input_payload)  # This should now send the request correctly
        return response
    except Exception as e:
        raise ValueError(f"Error raised by bedrock service: {e}")

# Main function for Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")   
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if user_question:  # Proceed only if the user has entered a question
        with st.spinner("Processing..."):
            # Load faiss_index with allow_dangerous_deserialization=True to handle deserialization security
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)  # Display the model's response to the user
            st.success("Done")

if __name__ == "__main__":
    main()
