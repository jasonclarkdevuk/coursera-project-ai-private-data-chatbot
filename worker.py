# Document processing and conversation management worker
# Part of the chatbot app that processes user messages and documents
# It uses LangChain library

import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate  # Updated import per deprecation notice
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxLLM

# Automatically detect and select the fastest available hardware to run your code
# Checks if a powerful GPU is available. Fallback on CPU if not
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None # RetrievalQA chain
chat_history = [] # To be able to hold history of the chat
llm_hub = None # Object for LLM
embeddings = None

# Function to initialise the LLM and its embeddings
def init_llm():
    global llm_hub, embeddings

    logger.info("Initializing WatsonxLLM and embeddings...")

    # Llama Model Configuration constants
    # Using an AI model from IBM watsonx
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID = "skills-network"

    # Model parameters
    model_parameters = {
        "max_new_tokens": 256, # Specify the exact maximum number of tokens the AI should generate in its response
        "temperature": 0.1, # Controls the randomness of the model's text generation
    }

    # Initialize Llama LLM using the updated WatsonxLLM API
    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=model_parameters
    )

    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # Initialize embeddings using a pre-trained model to represent the text data.
    # Embeddings are a way of translating human language into a format that a computer can actually understand (numbers, vectors etc)
    # Sentence Transformers are a family of models that are specially trained to look at text and understand their overall meaning
    # Create Hugging Face Embeddings
    # HuggingFaceEmbeddings takes text exactly as it is written and converts it into a vector based purely on the words in the text
    embeddings =  HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", # Fast and lightweight LLM model
        model_kwargs={"device": DEVICE} # Extra configuration settings
    )
    
    logger.debug("Embeddings initialized with model device: %s", DEVICE)

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    logger.info("Loading document from path: %s", document_path)

    # Load the document using PyPDFLoader library
    loader =  PyPDFLoader(document_path)
    documents = loader.load()

    logger.debug("Loaded %d document(s)", len(documents))

    # Split the document into chunks using RecursiveCharacterTextSplitter
    # Highly intelligent "smart-splitter" from the LangChain library
    # Tries to break down document into pieces - but tries to keep semantic meaning together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, # Maximum amount of characters per chunk
        chunk_overlap=64 # 64-character overlap buffer. Can help model keep context even if sentences are split in half
    )
    texts = text_splitter.split_documents(documents)

    logger.debug("Document split into %d text chunks", len(texts))

    # Create an Embeddings database using Chroma from the split text chunks
    # Chopped-up text is converted into numbers / vectors which is saved into a database system
    # Vector store creation - "specialized, searchable AI brain"
    logger.info("Initializing Chroma vector store from documents...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized.")

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Build the Retrieval Question Answer chain, which utilises the LLM and retriever for answering questions.
    # The chain uses the initialised LLM and the embeddings database to answer questions on the processed document
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff", # Tells LangChain to add all the retrieved document chunks directly into the prompt
        retriever=db.as_retriever(
            # Configures how the system searches the Chroma database
            # MMR = Maximum Marginal Relevance. Tries to find relevant but diverse chunks to ensure the AI gets a well-rounded view of the answer
            search_type="mmr", 
            search_kwargs={'k': 6, 'lambda_mult': 0.25}
        ),
        return_source_documents=False, # We just want the final text answer
        input_key="question" # Defines what the input variable name when we invoke the chain with a question later
        # chain_type_kwargs={"prompt": prompt}  # if you are using a prompt template, uncomment this part
    )

    logger.info("RetrievalQA chain created successfully.")
    
# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: %s", prompt)

    # Query the model using the new .invoke() method with a new prompt and current history
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})

    # LLM response to query
    answer = output["result"]

    logger.debug("Model response: %s", answer)

    # Update the chat history
    chat_history.append((prompt, answer))
    
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    # Return the model's response
    return answer

# Initialize the language model
init_llm()
logger.info("LLM and embeddings initialization complete.")
