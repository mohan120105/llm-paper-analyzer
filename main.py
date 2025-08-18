# main.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Define the path for the vector store
DB_FAISS_PATH = "vectorstores/db_faiss"

# Define a prompt template for the LLM
custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def create_rag_chain():
    """
    Creates the RAG (Retrieval-Augmented Generation) chain.
    """
    # Initialize the LLM
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

    # Load the embedding model (must be the same as in ingest.py)
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Load the vector store
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = db.as_retriever(search_kwargs={'k': 2}) # Retrieves the top 2 most relevant chunks

    # Create the prompt from the template
    prompt = set_custom_prompt()

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == '__main__':
    rag_chain = create_rag_chain()

    # Ask a question
    question = "What is the role of the gating network in a Mixture of Experts model?"
    print(f"Question: {question}")
    
    response = rag_chain.invoke(question)
    print("Answer:", response)