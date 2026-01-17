import os
from dotenv import load_dotenv
from pprint import pprint

from src.config import LLMConf, KnowledgeGraphConfig, EmbedderConf
from src.graph.graph_model import Ontology
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.graph_miner import GraphMiner
from src.schema import Chunk, ProcessedDocument

import os
import json
from dotenv import load_dotenv

# from typing import List, Optional, Dict
# from langchain_core.load.serializable import Serializable
# from pydantic import BaseModel, Field

from src.graph.graph_model import Ontology
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.local_ingestor import LocalIngestor
from src.ingestion.cleaner import Cleaner
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import ChunkEmbedder
from src.config import Source, ChunkerConf, LLMConf, KnowledgeGraphConfig, EmbedderConf
from src.ingestion.graph_miner import GraphMiner

env=load_dotenv("config.env", override=True)


kg_config = KnowledgeGraphConfig(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="vector"
)

chunker_conf = ChunkerConf(
    type="recursive",
    chunk_size=1000,
    chunk_overlap=100
)

llm_conf = LLMConf(
    model=os.getenv("AZURE_OPENAI_LLM_MODEL_NAME"),
    temperature=0,
    type="azure-openai",
    deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_LLM_VERSION"),
)

embedder_conf = EmbedderConf(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME"),
    type="azure-openai",
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_VERSION"),
)

ontology = Ontology(
    allowed_labels=["Person", "City", "Country", "Region"], 
    allowed_relations=["LIVES_IN", "BORN_IN"]
)

text= """ 
    Meet Marco Rossi, a proud native of Rome, Italy. With a passion for pasta and a love for football, 
    Marco embodies the vibrant spirit of his homeland. 
    Growing up amidst the ancient ruins and bustling piazzas of Rome, Marco developed 
    a deep appreciation for his city's rich history and cultural heritage. 
    From the iconic Colosseum to the majestic Vatican City, he has explored every corner of the Eternal City, 
    finding beauty in its timeless landmarks and hidden gems. 
    Italy, with its delectable cuisine and stunning landscapes, has always been Marco's playground. 
    From the rolling hills of Tuscany to the sparkling waters of the Amalfi Coast,
    he has embraced the diversity and beauty of his country, taking pride in its traditions and way of life.
"""

chunks = [
    "Meet Marco Rossi, a proud native of Rome, Italy. With a passion for pasta and a love for football, Marco embodies the vibrant spirit of his homeland. Growing up amidst the ancient ruins and bustling piazzas of Rome, Marco developed a deep appreciation for his city's rich history and cultural heritage.",
    "From the iconic Colosseum to the majestic Vatican City, he has explored every corner of the Eternal City, finding beauty in its timeless landmarks and hidden gems. Italy, with its delectable cuisine and stunning landscapes, has always been Marco's playground. From the rolling hills of Tuscany to the sparkling waters of the Amalfi Coast, he has embraced the diversity and beauty of his country, taking pride in its traditions and way of life."
]

doc = ProcessedDocument(
    filename="test.pdf", 
    source=text,
    chunks=[
        Chunk(
            chunk_id=1, 
            text=chunks[0], 
            chunk_size=500, 
            chunk_overlap=50,
            embeddings_model='mxbai-embed-large'
        ),
        Chunk(
            chunk_id=2, 
            text=chunks[1], 
            chunk_size=500, 
            chunk_overlap=50,
            embeddings_model='mxbai-embed-large'
        )
    ]
)

# embeddings
embedder = ChunkEmbedder(conf=embedder_conf)
doc = embedder.embed_document_chunks(doc)

# extract the knowledge graph
graph_miner = GraphMiner(conf=llm_conf, ontology=ontology)
doc= graph_miner.mine_graph_from_doc_chunks(doc)

doc.chunks[0].nodes
doc.chunks[0].relationships

knowledge_graph = KnowledgeGraph(
    conf=kg_config, 
    embeddings_model=embedder.embeddings
)

knowledge_graph._driver.verify_connectivity()

knowledge_graph._driver.verify_authentication()

knowledge_graph.store_chunks_for_doc(
    doc=doc
)

from src.agents.graph_qa import GraphAgentResponder
responder = GraphAgentResponder(
    qa_llm_conf=llm_conf, # TODO try different LLMs
    cypher_llm_conf=llm_conf,
    graph=knowledge_graph,
    rephrase_llm_conf=llm_conf
)