import json
import os
import sys
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

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Character split works better with this PDF dataset
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    # Create the Llama3 Model
    llm = Bedrock(model_id="us.meta.llama3-2-90b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    # Prepare the payload for invoking the Bedrock model
    payload = {
        "prompt": f"[INST] {query} [/INST]",
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
    return response_text

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

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            # Allow dangerous deserialization explicitly
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
