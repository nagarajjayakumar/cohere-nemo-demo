import glob
import os
import time
import logging

import cohere
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
import json

def load_environment_variables():
    """
    Load environment variables from a .env file.

    Returns:
        dict: A dictionary containing the loaded environment variables.
    """
    load_dotenv()
    return {
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_index": os.getenv("PINECONE_INDEX")
    }


def initialize_clients(cohere_api_key, pinecone_api_key):
    """
    Initialize Cohere and Pinecone clients.

    Args:
        cohere_api_key (str): The API key for Cohere.
        pinecone_api_key (str): The API key for Pinecone.

    Returns:
        tuple: A tuple containing the initialized Cohere and Pinecone clients.
    """
    cohere_client = cohere.Client(cohere_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    return cohere_client, pinecone_client


def create_embeddings(cohere_api_key):
    """
    Create a CohereEmbeddings object for generating embeddings.

    Args:
        cohere_api_key (str): The API key for Cohere.

    Returns:
        CohereEmbeddings: An initialized CohereEmbeddings object.
    """
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key
    )


def create_pinecone_collection(pinecone_client, index_name):
    """
    Create a Pinecone collection with the specified index name.

    Args:
        pinecone_client (Pinecone): The initialized Pinecone client.
        index_name (str): The name of the index to create or use.

    Returns:
        Pinecone.Index: The Pinecone index object.
    """
    try:
        logger.info(f"Creating 1024-dimensional index called '{index_name}'...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("Success")
    except:
        logger.info("Index already exists, continuing...")

    while not pinecone_client.describe_index(index_name).status['ready']:
        time.sleep(1)

    logger.info("Checking Pinecone for active indexes...")
    active_indexes = pinecone_client.list_indexes()
    print("Active indexes:", active_indexes)

    logger.info(f"Getting description for '{index_name}'...")
    index_description = pinecone_client.describe_index(index_name)
    print("Description:", index_description)

    logger.info(f"Getting '{index_name}' as object...")
    pinecone_index = pinecone_client.Index(index_name)
    logger.info("Success")

    return pinecone_index


def process_pdf_files(folder_path, embeddings, index_name):
    """
    Process PDF files in the specified folder and add them to the Pinecone index.

    Args:
        folder_path (str): The path to the folder containing PDF files.
        embeddings (CohereEmbeddings): The embeddings object for generating vector representations.
        index_name (str): The name of the Pinecone index to use.
    """
    pdf_files = glob.glob(os.path.join(folder_path, '**/*.pdf'), recursive=True)

    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        docsearch = LangchainPinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
        logger.info(f"Loaded PDF document {pdf_file} successfully to Pinecone index {index_name}")

def setup_logging(log_file='/tmp/app.log'):
    """Set up logging to both console and file."""
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(log_file, mode='a')  # File handler, 'a' for append mode

        # Create formatters and add it to handlers
        format = logging.Formatter('%(asctime)s - %(filename)s - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(format)
        f_handler.setFormatter(format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger
    except Exception as e:
        logger.info(f"Error setting up logging: {e}")
        return logging.getLogger(__name__)  # Return a default logger if setup fails


def dict_to_string(data):
    """
    Convert a dictionary to a formatted string for logging.

    Args:
        data (dict): The dictionary to convert.

    Returns:
        str: A formatted string representation of the dictionary.
    """
    return json.dumps(data, indent=2)


import os


def get_file_path(filename):
    """
    Get the full path of a file in the same directory as the current script.

    Args:
    filename (str): The name of the file.

    Returns:
    str: The full path to the file.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the file
    file_path = os.path.join(script_dir, filename)

    return file_path

# =========================
# Set up the logger
logger = setup_logging()