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
from src.agents.ontology_explorer import OntologyExplorer

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

# Chunking
chunker=Chunker(conf=ontology_chunker_conf)
ont_chunks=chunker.chunk_documents(docs_cleaned)
for i in range(len(docs)):
    print(f"Number of chunks in doc {i}: {len(ont_chunks[i].chunks)}")

# # Embedding
# embedder=ChunkEmbedder(conf=embedder_conf)
# docs_embeddings=embedder.embed_documents_chunks(docs_chunks)

####################################################################
# ONTOLOGY GENERATION
####################################################################

DOMAIN_DESCRIPTION="""
The domain focuses on the governance, management, and operational control of public sector finance within the UK Parliament, as codified in the Finance Rules Handbook V3.1.
It covers:

- Governance & Accountability: Structures, roles, and responsibilities for financial oversight, including committees, delegated authorities, and compliance frameworks.
- Budgeting & Planning: Annual and medium-term financial planning cycles, Estimates, budget setting, forecasting, and reporting mechanisms.
- Procurement & Expenditure: Rules and procedures for purchasing goods, services, and works; contract management; authorisation limits; and payment methods.
- Risk, Fraud & Internal Control: Policies for risk management, prevention and detection of fraud, loss management, and internal audit processes.
- Asset & Data Management: Safeguarding, recording, and disposal of assets; inventory control; insurance; and data security protocols.
- External Engagement: Procedures for engaging consultants, interims, and agency staff, including IR35 compliance and assurance requirements.
- Income, Debtors & Special Payments: Management of income streams, debtor processes, overpayments, losses, write-offs, and special payments.
- Transparency & Reporting: Requirements for financial disclosures, publication of expenditure, and annual reporting to Parliament and the public.

This domain provides a comprehensive framework for ensuring financial integrity, value for money, transparency, and accountability in the management of public resources within a parliamentary context.
"""

# LLM-driven ontology generation
ontology_explorer=OntologyExplorer(
    llm_conf, 
    domain_description=DOMAIN_DESCRIPTION
)

from src. factory.llm import fetch_llm
llm=fetch_llm(llm_conf)

ontology=ontology_explorer.find_suitable_ontology(docs=ont_chunks, pct_chunks=0.1, chunks_limit=1)
print(ontology.model_dump())

# # Save ontology
# try:
#     path=os.path.abspath(os.path.join(__file__, "../../assets/ontology.json"))
# except:
#     path=os.path.abspath(os.path.join(os.getcwd(), "../assets/ontology.json"))

# with open(path, "w", encoding="utf-8") as f:
#     f.write(json.dumps(ontology.model_dump()))