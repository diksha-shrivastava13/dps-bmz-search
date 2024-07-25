import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# models
embed_model = OpenAIEmbedding(embed_batch_size=10)
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0,
                  api_key=az_openai_api_key, azure_endpoint=az_openai_endpoint)

# Configurations
Settings.embed_model = embed_model
Settings.llm = llm


def search_result(file_path, user_query):
    loader = SimpleDirectoryReader(input_files=[file_path], required_exts=[".pdf"])
    docs = loader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    query_response = query_engine.query(user_query + " Make the answer as detailed and as comprehensive as required."
                                        + " Make sure to use the documents as context to answer the question. "
                                        + "If you cannot find an answer from the documents, tell the user to go through"
                                          "the original document they uploaded")
    return query_response

# need a filepath and user query to return a string
