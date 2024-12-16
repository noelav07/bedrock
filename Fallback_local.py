import json
import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

def data_ingestion():
    documents = []

    if any(filename.endswith(".pdf") for filename in os.listdir("data")):
        pdf_loader = PyPDFDirectoryLoader("data")
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)

    if any(filename.endswith(".txt") for filename in os.listdir("data")):
        text_documents = []
        for filename in os.listdir("data"):
            if filename.endswith(".txt"):
                with open(os.path.join("data", filename), "r") as file:
                    content = file.read()
                    text_documents.append(content)
        documents.extend(text_documents)

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    llm = Bedrock(model_id="us.meta.llama3-2-90b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

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

def get_response_llm(llm, vectorstore_faiss, query, model_db_vectorstore=None):
    local_docs = vectorstore_faiss.similarity_search(query, k=3)
    local_context = "\n".join([doc.page_content for doc in local_docs])

    if model_db_vectorstore:
        model_db_docs = model_db_vectorstore.similarity_search(query, k=3)
        model_db_context = "\n".join([doc.page_content for doc in model_db_docs])
        combined_context = local_context + "\n" + model_db_context
    else:
        combined_context = local_context

    prompt = PROMPT.format(context=combined_context, question=query)

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
    
    return response_text.strip()

def get_fallback_response(query):
    # Fallback to external knowledge or model's pre-trained knowledge (if local data doesn't work)
    llm = get_llama3_llm()
    prompt = f"Question: {query}\nAnswer:"
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
    
    return response_text.strip()

def main():
    st.set_page_config("Document Q&A")

    st.header("Document Q&A")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update and fetch data:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()  
                get_vector_store(docs)  
                st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()

            # Try to get the response from local data first
            local_response = get_response_llm(llm, faiss_index, user_question)

            if not local_response or "I don't know" in local_response:
                # If local data doesn't provide a valid answer, fallback to external knowledge
                st.write("Fetching from external knowledge...")
                external_response = get_fallback_response(user_question)
                st.write(external_response)
            else:
                st.write(local_response)

            st.success("Done")

if __name__ == "__main__":
    main()
