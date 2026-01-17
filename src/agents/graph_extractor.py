from src.utils.logger import get_logger
from typing import Optional

# from langchain_neo4j.graphs.graph_document import Relationship, Node
from langchain_core.documents import Document

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology, _Graph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()

        self.prompt.partial_variables = {
            'allowed_labels':ontology.allowed_labels if ontology and ontology.allowed_labels else "", 
            'labels_descriptions': ontology.labels_descriptions if ontology and ontology.labels_descriptions else "", 
            'allowed_relationships': ontology.allowed_relations if ontology and ontology.allowed_relations else ""
        }


    def extract_graph(self, text: str) -> _Graph:
        """ 
        Extracts a graph from a text.
        """
        input_prompt=self.prompt.format(input_text=text)
        if self.llm is not None:
            try:
                raw=self.llm.chat.completions.parse(
                messages=[
                {
                    "role": "system",
                    "content": "You are a top-tier algorithm designed for extracting information in structured formats to build a Knowledge Graph."
                },
                {
                    "role": "user",
                    "content": input_prompt
                }
                ],
                model="gpt-5.2",
                max_completion_tokens=20000,
                response_format=_Graph
                )
                graph=raw.choices[0].message.parsed
                return graph 
                
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")