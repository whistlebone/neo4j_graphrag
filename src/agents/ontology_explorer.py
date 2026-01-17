import random
from src.utils.logger import get_logger
from typing import Optional, List

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology
from src.prompts.ontology_explorer import get_ontology_creation_prompt
from src.schema import ProcessedDocument

logger = get_logger(__name__)

class OntologyExplorer:
    """ 
    Agent in charge to discover the best `Ontology` given a list of Chunks from 
    documents from the same domain. 
    """

    def __init__(self, llm_conf: LLMConf, domain_description: Optional[str]=None):
        self.llm = fetch_llm(llm_conf)
        self.prompt = get_ontology_creation_prompt()

        if domain_description is not None: 
            self.prompt.partial_variables = {"domain_description": domain_description}


    def find_suitable_ontology(
            self,
            docs: List[ProcessedDocument], 
            pct_chunks: float=0.1,
            chunks_limit: int=10
        ) -> Ontology:
        """
        Passes a (random) list of chunks from the documents to the LLM, so that they are used by the model
        to infer an `Ontology` for the `KnowledgeGraph`  

        -------
        params:
        -------
        `docs`: `List[ProcessedDocument]`
            List of already chunked Documents 
        `pct_chunks`: `float`
            Percentage of chunks from each Document to be passed to the prompt
        `chunks_limit`: `int`
            Maximum number of chunks to pass in the prompt
        """
        
        context_chunks = []

        for doc in docs: 
            sampled_chunks = random.sample(doc.chunks, round(len(doc.chunks)*pct_chunks))
            
            for chunk in sampled_chunks:
                context_chunks.append(chunk.text)

        if len(context_chunks) > chunks_limit:
            context_chunks = random.sample(context_chunks, chunks_limit)

        input_prompt=self.prompt.format(texts=context_chunks)

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
            response_format=Ontology
            )
            ontology=raw.choices[0].message.parsed
            return ontology

        except Exception as e:
            logger.warning(f"Unable to find an Ontology with error: {e}")