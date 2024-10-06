from typing import List

from langchain_cohere import CohereEmbeddings
from nemoguardrails import LLMRails
from nemoguardrails.embeddings.basic import EmbeddingModel
from dotenv import load_dotenv
import os

load_dotenv()

class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model."""

    engine_name = "cohere"

    #embedding_model="embed-english-v3.0", api_key=env_vars["cohere_api_key"]
    def __init__(self, embedding_model: str = "embed-english-v3.0"):
        """Initialize the Cohere embedding model."""
        #super().__init__()
        print("initating embedding model")
        self.cohere_embeddings = CohereEmbeddings(
            model=embedding_model,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode the given documents asynchronously."""
        return self.cohere_embeddings.embed_documents(documents)

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode the given documents."""
        return self.cohere_embeddings.embed_documents(documents)

def init(app: LLMRails):
    print("init LLMRails model")
    app.register_embedding_provider(CohereEmbeddingModel, "cohere")
