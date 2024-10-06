import asyncio
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from nemoguardrails import RailsConfig, LLMRails
from config import CohereEmbeddingModel


model = ChatCohere(
        model="command-r-plus",
        temperature=0,
    )

async def generate_help_offer(context: str):
    print("inside generate_help_offer")
    try:
        prompt = f"""
        Based on the following context, generate a helpful offer for further assistance:

        Context: {context}

        The offer should be friendly, relevant to the context, and encourage further engagement on the topic of deep learning and programming in the context of deep learning.
        """

        messages = [[HumanMessage(content=prompt)]]
        help_response = await model.agenerate(messages)
        help_offer = help_response.generations[0][0].text
        print(f"Generated help offer: {help_offer}")
        return help_offer
    except Exception as e:
        print(f"Error in generate_help_offer: {str(e)}")
        return "Is there anything else I can help you with regarding deep learning or programming in the context of deep learning?"

async def func(inputs: str):
    print("inside the FUNC")
    return "hello you are an orange"


colang_content = """
# define limits
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm a shopping assistant, I don't like to talk of politics."

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
  

"""
yaml_content = """
models:
 - type: main
   engine: cohere
   model: command-r-plus
"""

# initialize rails config
config = RailsConfig.from_content(
    colang_content=colang_content,
    yaml_content=yaml_content
)
# create rails
rails = LLMRails(config, verbose=True)
rails.register_embedding_provider(CohereEmbeddingModel, "cohere")
rails.register_action(action=func, name="response")
rails.register_action(action =generate_help_offer,name = "generate_help_offer")

async def process_query():
    response = await rails.generate_async(prompt="what is deep learning?")
    return response

# Create a main function to run the asynchronous code
async def main():
    # Add this line to build the index
    result = await process_query()
    print(result)


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
# Run the main function
load_environment_variables()
asyncio.run(main())