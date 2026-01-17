print(f"__file__: {__file__}")
print(f"__package__: {__package__}")

####################################################################
# IMPORT LIBRARIES AND SET UP CONFIG
####################################################################

# Import libraries
import os
import json
from dotenv import load_dotenv

from src.config import Source, ChunkerConf, LLMConf, KnowledgeGraphConfig, EmbedderConf
from src.ingestion.local_ingestor import LocalIngestor
from src.ingestion.cleaner import Cleaner
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import ChunkEmbedder
from src.graph.graph_model import Ontology
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.graph_miner import GraphMiner

load_dotenv('config.env', override=True)

# Knowledge graph config
kg_config = KnowledgeGraphConfig(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name=os.getenv("INDEX_NAME")
)

# Ontology chunker config
ontology_chunker_conf = ChunkerConf(
    type="recursive",
    chunk_size=os.getenv("ONTOLOGY_CHUNKSIZE"),
    chunk_overlap=os.getenv("ONTOLOGY_OVERLAP")
)

# Graph embeddings chunker config
graph_embed_conf = ChunkerConf(
    type="recursive",
    chunk_size=os.getenv("GRAPH_CHUNKSIZE"),
    chunk_overlap=os.getenv("GRAPH_OVERLAP")
)

# Graph model config
llm_conf = LLMConf(
    type="azure-openai",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_TEXT_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_TEXT_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_TEXT_MODEL_NAME"),
    deployment=os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_TEXT_API_VERSION"),
)

# Chat model config
chat_model_conf = LLMConf(
    type="azure-openai",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME"),
    deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
)

# Embedding model config
embedder_conf = EmbedderConf(
    type="azure-openai",
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_VERSION"),
)

####################################################################
# DATA INGESTION, CLEANING AND CHUNKING
####################################################################

# Source data folder
source=Source(folder="raw_data")

# Load from local folder
ingestor=LocalIngestor(source=source)
docs=ingestor.batch_ingest()

# Clean docs
rem_start=5477
rem_end=-1
cleaner=Cleaner()
docs_cleaned=cleaner.clean_documents(docs)
docs_cleaned[0].source=docs_cleaned[0].source[rem_start:rem_end] # Remove start and end characters e.g. Contents, Appendix, References

# Generate graph chunks and embed
chunker=Chunker(conf=graph_embed_conf)
graph_chunks=chunker.chunk_documents(docs_cleaned)
for i in range(len(docs)):
    print(f"Number of chunks in doc {i}: {len(graph_chunks[i].chunks)}")

# Embedding
embedder=ChunkEmbedder(conf=embedder_conf)
docs_embeddings=embedder.embed_documents_chunks(graph_chunks)

####################################################################
# IMPORT EXISTING ONTOLOGY
####################################################################

# Load existing ontology if it exists
try:
    path=os.path.abspath(os.path.join(__file__, "../assets/ontology.json"))
except:
    path=os.path.abspath(os.path.join(os.getcwd(), "assets/ontology.json"))

try:
    os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        ont_json=json.load(f)
except:
    print("Ontology does not exist")

ontology=Ontology(
    allowed_labels=ont_json["allowed_labels"], 
    labels_descriptions=ont_json["labels_descriptions"],
    allowed_relations=ont_json["allowed_relations"]
)

####################################################################
# EXTRACT GRAPH COMPONENTS
####################################################################

# Mine graph nodes and edges
graph_miner=GraphMiner(
    conf=llm_conf, 
    ontology=ontology
)
graph_components=graph_miner.mine_graph_from_doc_chunks(docs_embeddings[0])

i=100
print(f"Nodes identified in chunk {i}: {graph_components.chunks[0].nodes}")
print(f"Relationships identified in chunk {i}: {graph_components.chunks[0].relationships}")

# Connect to neo4j graph instance
knowledge_graph=KnowledgeGraph(
    conf=kg_config, 
    embeddings_model=embedder.embeddings
)
knowledge_graph._driver.verify_connectivity()
knowledge_graph._driver.verify_authentication()

# Check number of nodes and edges
print(f"Number of nodes: {knowledge_graph.number_of_labels}")
print(f"Number of edges: {knowledge_graph.number_of_relationships}")
print(f"Name of indexer: {knowledge_graph.index_name}")

# Create knowledge graph
knowledge_graph.store_chunks_for_doc(
    doc=graph_components
)