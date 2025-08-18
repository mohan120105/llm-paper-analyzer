# agent.py
import os
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- 1. Define the State for our graph ---
# This is the object that will be passed between nodes
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

# --- 2. Define the nodes for our graph ---

# Node 1: Retrieve Documents
def retrieve_documents(state):
    """
    Retrieves documents from the FAISS vector store based on the question.
    """
    print("---NODE: RETRIEVING DOCUMENTS---")
    question = state["question"]
    
    # Load the embedding model and vector store
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local("vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve documents
    retriever = db.as_retriever(search_kwargs={'k': 3})
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}
# COPY THIS ENTIRE CORRECTED FUNCTION
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---NODE: GRADING DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # This is the corrected line:
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question", "context"],
    )

    parser = JsonOutputParser()
    
    chain = prompt | llm | parser

    for doc in documents:
        # Check each document for relevance
        score = chain.invoke({"question": question, "context": doc.page_content})
        grade = score.get("score")
        if grade.lower() == "yes":
            # If any document is relevant, we are good to go
            print("---DECISION: RELEVANT DOCUMENTS FOUND---")
            return {"documents": documents, "question": question}

    # If no relevant documents are found
    print("---DECISION: NO RELEVANT DOCUMENTS FOUND---")
    return {"documents": [], "question": question}


# Node 3: Generate the Answer
def generate_answer(state):
    """
    Generates an answer using the retrieved documents.
    """
    print("---NODE: GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]

    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    

    prompt = PromptTemplate(
        template="""Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}

# --- 3. Define the conditional edge for routing ---

def decide_to_generate(state):
    """
    Determines whether to generate an answer or end the process.
    """
    print("---CONDITIONAL EDGE: DECIDING TO GENERATE---")
    documents = state["documents"]
    
    if not documents:
        # If there are no relevant documents, end the process
        print("---DECISION: ENDING PROCESS---")
        return END
    else:
        # Otherwise, generate an answer
        print("---DECISION: PROCEEDING TO GENERATE---")
        return "generate"

# --- 4. Build the graph ---

workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer)

# Set the entry point
workflow.set_entry_point("retrieve")

# Add the edges
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        END: END,
    },
)
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
app = workflow.compile()

# --- 5. Run the agent ---

if __name__ == "__main__":
    # Test with a relevant question
    print("--- RUNNING WITH RELEVANT QUESTION ---")
    inputs = {"question": "What is the role of the gating network in a Mixture of Experts model?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
    print("\nFINAL GENERATION:")
    print(value["generation"])
    print("\n" + "="*50 + "\n")

    # Test with an irrelevant question
    print("--- RUNNING WITH IRRELEVANT QUESTION ---")
    inputs = {"question": "What is the capital of France?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
    # Handle the case where generation might not happen
    if "generation" in value:
        print("\nFINAL GENERATION:")
        print(value["generation"])
    else:
        print("\nFINAL RESPONSE: No relevant documents found to answer the question.")