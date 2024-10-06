from dotenv import load_dotenv

from utils.util import (load_environment_variables,
                        initialize_clients,
                        create_embeddings,
                        create_pinecone_collection, process_pdf_files)


# Initialize the logger
#logger = setup_logging()

def main():
    """
    The main function that orchestrates the entire process of loading environment variables,
    initializing clients, creating embeddings, setting up the Pinecone collection,
    and processing PDF files.
    """
    load_dotenv()

    print("loading pinecone index ")
    env_vars = load_environment_variables()
    cohere_client, pinecone_client = initialize_clients(env_vars["cohere_api_key"], env_vars["pinecone_api_key"])
    embeddings = create_embeddings(env_vars["cohere_api_key"])

    index = create_pinecone_collection(pinecone_client, env_vars["pinecone_index"])

    folder_path = '//docs/'
    #process_pdf_files(folder_path, embeddings, env_vars["pinecone_index"])
    print("process pinecone index is completed !!! ")


if __name__ == '__main__':
    main()
