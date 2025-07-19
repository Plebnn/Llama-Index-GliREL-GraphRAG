import json
from llama_index.core import Document
from src.GlirelPathExtractor import GlirelPathExtractor 
from src.RecursivePathExtractor import RecursiveLLMPathExtractor
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex,Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import openlit
# initialize openlit for monitoring
#openlit.init(
#  otlp_endpoint="http://127.0.0.1:4318",
#  application_name="hybrid",
#  environment="default")
# define LLM connection
llm = Ollama(
    model= "gemma3:12b",
    request_timeout=120.0,
    context_window=8128,
    temperature=0.0)
Settings.llm = llm
Settings.chunk_size=512
Settings.chunk_overlap=64
embed_model = OllamaEmbedding(
    model_name="snowflake-arctic-embed2:latest",
    ollama_additional_kwargs={"mirostat": 0},)
Settings.embed_model = embed_model
with open('../.data/novel.json', 'r') as file:
    data = json.load(file)
documents = [Document(text=t["context"],id_=t["corpus_name"]) for t in data]
#define Extractors
extractorGli = GlirelPathExtractor(device="cuda")
extractorLLM = RecursiveLLMPathExtractor(llm=Settings.llm, num_workers=1, max_paths_per_chunk=15)
# index
for document in documents:
    index = PropertyGraphIndex.from_documents(
        documents=[document],
         # use extractorGli for GliNER GliRel pipline triplet extraction, LLM for LLM based and both for hybrid
        kg_extractors=[
            extractorGli,
            extractorLLM
            ],
        use_async = False,)
    novel_id = document.id_
    index.storage_context.persist(persist_dir=f"./.persistent_storage/hybrid/{novel_id}")
