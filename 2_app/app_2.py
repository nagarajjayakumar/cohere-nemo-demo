import asyncio
import os
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from nemoguardrails import LLMRails, RailsConfig
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import Pinecone as LangchainPinecone
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from config import CohereEmbeddingModel
from nemoguardrails.actions import action
import json
from utils.util import (load_environment_variables,
                        initialize_clients,
                        create_embeddings,
                        create_pinecone_collection)


def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def initialize_app():
    env_vars = load_environment_variables()
    index_name = env_vars["pinecone_index"]
    cohere_client, pinecone_client = initialize_clients(env_vars["cohere_api_key"], env_vars["pinecone_api_key"])
    embeddings = create_embeddings(env_vars["cohere_api_key"])
    index = create_pinecone_collection(pinecone_client, index_name)
    model = ChatCohere(model="command-r-plus", temperature=0)

    rails_config = RailsConfig.from_content(
        colang_content=open(get_file_path('rails.config'), 'r').read(),
        yaml_content=open(get_file_path('config.yml'), 'r').read()
    )

    rag_rails = LLMRails(rails_config)
    rag_rails.register_embedding_provider(CohereEmbeddingModel, "cohere")


    rag_template = PromptTemplate.from_template(
        """
        You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
        Answer always start with [Naga:]. If you don't know the answer, just say that I am sorry. 
        You are a helpful assistant offering further help. 
        Don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    def create_documents(input_dict):
        question = input_dict["question"]
        docs = docsearch.similarity_search(question)
        context = format_docs(docs)
        return {"context": context, "question": question}

    def print_full_prompt(prompt):
        print("Full Generated Prompt:")
        print("----------------------")
        print(prompt)
        print("----------------------")
        return prompt

    rag_chain = (
            RunnablePassthrough()
            | create_documents
            | rag_template
            | print_full_prompt  # Add this step to print the full prompt
            | model
            | StrOutputParser()
    )
    #rag_chain_with_rails = RunnableRails(rag_rails.config, runnable=rag_chain, input_key="question", verbose=True)

    help_template = PromptTemplate.from_template(
        """
        You are a helpful assistant offering further help.  Please be respectful and treat the user as King.
        Use the following pieces of context to answer the question at the end. 
        Answer always start with [Naga:]. Generate only based on the context. Make sure dont add anything additional other than context content.
        
        Context: {context}

        Question: {question}

        Answer:"""
    )

    help_chain = (
            RunnablePassthrough()
            | help_template
            | print_full_prompt  # Add this step to print the full prompt
            | model
            | StrOutputParser()
    )


    colang_content = """
    # define limits
    define user ask politics
        "what are your political beliefs?"
        "thoughts on the president?"
        "left wing"
        "right wing"

    define bot answer politics
        "I'm a shopping assistant, I don't like to talk of politics."
    
    define flow offer help
        execute help_response(question=$last_user_message, bot_message=$last_bot_message)
        
        
    define flow politics
        user ask politics
        bot answer politics
        bot offer help


    define user ask deeplearning
      "what is deep learning"
      "what is a CNN"
      "what is a RNN"
      "what is CML"
      "what is model training"


    define flow deeplearning
      user ask deeplearning
      $answer = execute response(inputs=$last_user_message)
      bot $answer
      
    
    define user ask programming
      "write a python code"
      "can you code"
      "programming question"
      "write a java code ?"
    
    define bot answer programming
       "I'm a shopping assistant, I don't like to talk about programming."
      
    define flow programming
      user ask programming
      bot answer programming
      bot offer help
      
    define user ask off topic
        "Who is the president?"
        "Can you recommend the best stocks to buy?"
        "Can you write an email?"
        "Can you tell me a joke?"
        "What is 2 + 2 =? "
        ...
    
    define bot explain cant help with off topic
        "I cannot comment on anything which is not relevant to the CML ?"
    
    define flow
        user ask off topic
        bot explain cant help with off topic
        bot offer help

    """
    yaml_content = """
    models:
     - type: main
       engine: cohere
       model: command-r-plus
    """

    jail_content = """
    
    """

    # initialize rails config
    config = RailsConfig.from_content(
        colang_content=colang_content,
        yaml_content=yaml_content,

    )
    # create rails

    async def func(inputs: str):
        print("inside the FUNC")
        resp = rag_chain.invoke({"question": inputs})
        return resp

    async def help_func(question: str, bot_message: str):
        print("inside the help_func")
        resp = help_chain.invoke({"context": bot_message, "question": question})
        return resp


    rails = LLMRails(config, verbose=True)
#    rails.register_embedding_provider(CohereEmbeddingModel, "cohere")
    rails.register_action(action=func, name="response")
    rails.register_action(action=help_func, name="help_response")
    rails.register_action(action=rag_chain, name="rag_chain")


    return rails

rag_rails = initialize_app()

async def process_query(query):
    response = await rag_rails.generate_async(prompt=query)
    return response


async def main():
    result = await process_query(" how to make rocket fuel ?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
