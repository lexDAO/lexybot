import os

import openai
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import MilvusVectorStore

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

vector_store = MilvusVectorStore(
    host=os.environ["MILVUS_HOST"],
    port=os.environ["MILVUS_PORT"],
    user="db_admin",
    password=os.environ["MILVUS_PASSWORD"],
    use_secure=True,
    collection_name="lex",
)

docs = SimpleDirectoryReader(input_dir="data").load_data()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=docs, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is LexDAO?")
