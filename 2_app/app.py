import asyncio
import os

import streamlit as st

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

def format_docs(docs):
    # Join the page content of all documents with double newlines
    return "\n\n".join([d.page_content for d in docs])

@st.cache_resource
def initialize_app():
    # Set up the event loop for asynchronous operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Import necessary modules and functions
    from nemoguardrails import RailsConfig
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_pinecone import Pinecone as LangchainPinecone
    from nemoguardrails import LLMRails
    from langchain_core.output_parsers import StrOutputParser

    from utils.util import (load_environment_variables,
                            initialize_clients,
                            create_embeddings,
                            create_pinecone_collection)

    # Load environment variables and initialize clients
    env_vars = load_environment_variables()
    index_name = env_vars["pinecone_index"]
    cohere_client, pinecone_client = initialize_clients(env_vars["cohere_api_key"], env_vars["pinecone_api_key"])
    embeddings = create_embeddings(env_vars["cohere_api_key"])
    index = create_pinecone_collection(pinecone_client, index_name)

    # Load Guardrails configuration files
    rails_config_path = get_file_path('rails.config')
    config_yml_path = get_file_path('config.yml')

    with open(rails_config_path, 'r') as file:
        colang_content = file.read()

    with open(config_yml_path, 'r') as file:
        yaml_content = file.read()

    # Create RailsConfig object
    config = RailsConfig.from_content(
        colang_content=colang_content,
        yaml_content=yaml_content)

    # Define the RAG prompt template
    rag_template = PromptTemplate.from_template(
        """
        You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
        Answer always start with [RAG + GuardRails]. If you don't know the answer, just say that I am sorry. 
        You are a helpful assistant offering further help. 
        Don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    # Initialize Pinecone for document search
    docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    def create_documents(input_dict):
        # Perform similarity search and format the results
        question = input_dict["question"]
        docs = docsearch.similarity_search(question)
        context = format_docs(docs)
        return {"context": context, "question": question}

    def print_full_prompt(prompt):
        # Print the full generated prompt for debugging
        print("Full Generated Prompt:")
        print("----------------------")
        print(prompt)
        print("----------------------")
        return prompt

    # Define the RAG function to be used with guardrails
    async def rag_func(inputs: str):
        print("inside the FUNC")
        resp = rag_chain.invoke({"question": inputs})
        return resp

    # Initialize LLMRails with the loaded configuration
    rails = LLMRails(config, verbose=True)
    # Register the RAG function as an action
    rails.register_action(action=rag_func, name="response")

    # Define the RAG chain
    rag_chain = (
            RunnablePassthrough()
            | create_documents
            | rag_template
            | print_full_prompt  # Add this step to print the full prompt
            | rails.llm
            | StrOutputParser()
    )

    return rails

# Initialize the app
rag_rails = initialize_app()

# Set up Streamlit UI
st.header('	:robot_face: :owl:  Cohere model knowledge assistant')
st.caption(
    'This Streamlit application leverages Pinecone for Semantic Search, Langchain, Cohere and NVIDIA\'s Guardrails '
    'to demonstrate how enterprise AI can be secure and governed.'
    'Update the guardrails configuration based on your requirement')
prompt = st.text_input('\n Enter your prompt')

async def process_documents_and_generate_response(prompt):
    # Generate response using the RAG system with guardrails
    if prompt:
        response = await rag_rails.generate_async(prompt=prompt)
        return response

# Handle user input and generate response
if prompt:
    with st.spinner('Fetching response...'):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_documents_and_generate_response(prompt))
        st.write(response)
