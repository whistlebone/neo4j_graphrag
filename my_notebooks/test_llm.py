import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology
from langchain_core.output_parsers import StrOutputParser
load_dotenv("config.env", override=True)

client=AzureOpenAI(
    api_key          = os.getenv(f"AZURE_OPENAI_TEXT_API_KEY"),
    azure_endpoint   = os.getenv(f"AZURE_OPENAI_TEXT_ENDPOINT"),
    azure_deployment = os.getenv(f"AZURE_OPENAI_TEXT_DEPLOYMENT_NAME"),
    api_version      = os.getenv(f"AZURE_OPENAI_TEXT_API_VERSION")
)

######################################
# TESTING
######################################

prompt= """
You are a top-tier algorithm designed for extracting information in structured formats to build a Knowledge Graph.

You will be given a description of the domain and a list of texts from the same domain: 
your task is to use them as context to create an `Ontology`, which is a representation 
of knowledge in the form of allowed labels (and their descriptions) as well as allowed relationships.

Remember that an Ontology is defined as:  

```
class Ontology: 
    '''
    Used to describe arbitrary, project-specific allowed labels and relationships.
    '''
    allowed_labels: List[str]
    labels_descriptions: Dict[str, str]
    allowed_relations: List[str]
```

You MUST return ONLY the Ontology inferred from the context given to you. Do not add anything else.
If the description of the domain is empty, just use the context texts as a source to infer the Ontology. 

### BEGIN! 
CONTEXT TEXTS: {text}
"""

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

raw=client.chat.completions.create(

    messages=[
        {
            "role": "user",
            "content": prompt.format(text=text)
        }
    ],
    model=os.getenv("AZURE_OPENAI_TEXT_MODEL_NAME"),
    max_completion_tokens=20000
)

raw=client.chat.completions.parse(
    messages=[
        {
            "role": "user",
            "content": prompt.format(text=text)
        }
    ],
    model=os.getenv("AZURE_OPENAI_TEXT_MODEL_NAME"),
    max_completion_tokens=20000,
    response_format=Ontology
)

raw.choices[0].message.parsed