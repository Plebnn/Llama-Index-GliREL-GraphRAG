import asyncio
from typing import Any, Callable, Optional, Sequence, Union, List, Tuple
import json
import re
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core import Settings
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate

from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core import Settings

from prompts import DEFAULT_ENTITIES, DEFAULT_RELATIONS, DEFAULT_RECURSIVE_START_EXTRACT_PROMPT, DEFAULT_RECURSIVE_CHECK_PROMPT, DEFAULT_RECURSIVE_LOOP_EXTRACT_PROMPT

def default_parse_recursive_triplets(
    llm_output: str,
):
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.

    """
    #triplets = []
    entities = []
    relations = []
    try:
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head")
            head_type = item.get("head_type")
            relation = item.get("relation")
            tail = item.get("tail")
            tail_type = item.get("tail_type")
            
            
            #head_node = EntityNode(name=head, label=head_type)
            #tail_node = EntityNode(name=tail, label=tail_type)
            #relation_node = Relation(source_id=head_node.id, target_id=tail_node.id, label=relation)
            #triplets.append((head_node, relation_node, tail_node))            
            entities.append((head,head_type))
            entities.append((tail,tail_type))
            relations.append((head,relation,tail))

    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, relation, tail, and tail_type
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\'],\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)
        
        for match in matches:
            head, head_type, relation, tail, tail_type = match
            #head_node = EntityNode(name=head, label=head_type)
            #tail_node = EntityNode(name=tail, label=tail_type)
            #relation_node = Relation(source_id=head_node.id, target_id=tail_node.id, label=relation)
            #triplets.append((head_node, relation_node, tail_node))            
            entities.append((head,head_type))
            entities.append((tail,tail_type))
            relations.append((head,relation,tail))
    return entities,relations

def relations_dict_to_list(base_dict:dict):
    if base_dict == None:
        return None
    return list(base_dict.keys())
    

class RecursiveLLMPathExtractor(TransformComponent):
    """
    Extract triples from a graph.

    Uses an LLM and a prompt to extract knowledge triples with a BEGINNING_EXTRACTION_PROMPT
      and recursively checks with CHECK_PROMPT if there are any extrations missing, if YES the LOOP_EXTRACTION_PROMPT is called. 
    If initialized first in the pipeline the triples will be generated with a BEGINNING_EXTRACTION_PROPMT.
    If any other Extractors are called first -> triples allready exists, CHECK_PROMPT is called to see if another extraction loop is nessecary

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.

    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    max_loops: int
    allowed_entity_types: List[str]
    allowed_relation_types: dict



    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_recursive_triplets,
        max_paths_per_chunk: int = 10,
        max_loops: int = 4,
        num_workers: int = 4,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relation_types: Optional[dict] = None
    ) -> None:
        """Init params."""
        

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_RECURSIVE_START_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
            max_loops=max_loops,
            allowed_entity_types= allowed_entity_types or DEFAULT_ENTITIES,
            allowed_relation_types = allowed_relation_types or DEFAULT_RELATIONS
        )

    @classmethod
    def class_name(cls) -> str:
        return "SimpleLLMPathExtractor"

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _start_extraction(self, text, prompt):
        try:
            llm_response = await self.llm.apredict(
            prompt,
            text=text,
            max_knowledge_triplets=self.max_paths_per_chunk,
            allowed_entity_types=", ".join(self.allowed_entity_types)
            if len(self.allowed_entity_types or []) > 0
            else "No entity types provided, You are free to define them.",
            allowed_relation_types=", ".join(self.allowed_relation_types or [])
            if len(self.allowed_relation_types or []) > 0
            else "No relation types provided, You are free to define them.",
        )
            entities,relations = self.parse_fn(llm_response)
        except ValueError:
            entities=[]
            relations=[]
    
        
        return set(entities), set(relations)
    
    async def _check_if_complete(self, text, prompt, existing_nodes, existing_relations) -> bool:
        try:
            
            llm_response = await self.llm.apredict(
            prompt,
            text=text,
            exisiting_nodes= str(existing_nodes),
            exisiting_relations=str(existing_relations)
            )
            
            if "yes" in llm_response.lower():
                return True
            return False
        except ValueError:
            print("Value error in complete check")
            return False

    async def _loop_extraction(self, text, prompt,existing_nodes, existing_relations):
        try:

            

            llm_response = await self.llm.apredict(
            prompt,
            text=text,
            max_knowledge_triplets=self.max_paths_per_chunk,
            existing_nodes = existing_nodes,
            existing_relations = existing_relations,
            allowed_entity_types=", ".join(self.allowed_entity_types)
            if len(self.allowed_entity_types or []) > 0
            else "No entity types provided, You are free to define them.",
            allowed_relation_types=", ".join(self.allowed_relation_types or [])
            if len(self.allowed_relation_types or []) > 0
            else "No relation types provided, You are free to define them."
            
        )
            entities, relations = self.parse_fn(llm_response) 
        except ValueError:
            entities=[]
            relations=[]
    
        
        return set(entities), set(relations)

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

    
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])    
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        
        unique_entities= []
        unique_relations =[]

        if existing_nodes == [] or existing_relations == []: # condition, too little 
            unique_entities, unique_relations = await self._start_extraction(
                text=text,
                prompt=self.extract_prompt)
        else:
            print("exisisting relations and entities have been found, skip first extraction loop")
            for entity in existing_nodes:
                unique_entities.append([entity.name, entity.label])
            for relation in existing_relations:
                unique_relations.append([relation.source_id,relation.label,relation.target_id])


        extraction_is_complete = await self._check_if_complete(text=text, prompt=DEFAULT_RECURSIVE_CHECK_PROMPT,existing_nodes=unique_entities, existing_relations=unique_relations)
        print(f"Is extraction complete? {extraction_is_complete}")
        count=0
        while count <= self.max_loops and not extraction_is_complete:
            count = count+ 1
            unique_entities, unique_relations = await self._loop_extraction(
                text=text,
                existing_nodes=unique_entities,
                existing_relations= unique_relations,
                prompt=DEFAULT_RECURSIVE_LOOP_EXTRACT_PROMPT)
            
            extraction_is_complete = await self._check_if_complete(text=text, prompt=DEFAULT_RECURSIVE_CHECK_PROMPT,existing_nodes=existing_nodes, existing_relations=existing_relations)
            if extraction_is_complete:
                print("All nessecary entities have been found for chunk")
            if len(unique_relations) >= self.max_paths_per_chunk:
                print("max number of relations per chunck found")
                extraction_is_complete = True
        
        if unique_entities != []:
            unique_entities = set(unique_entities)
        if unique_entities:
                for entity in unique_entities:
                    existing_nodes.append(
                        EntityNode(
                            name=entity[0], 
                            label=entity[1], 
                            properties={}   
                        )
            )
        else:
            print("NO ENTITIES FOUND FOR THIS NODE!!!")

        if unique_relations != []:
            unique_relations = set(unique_relations)
        if unique_relations:                   # relation looks like this: ('Obama', 'received', 'Ripple of Hope Award')
            for relation in unique_relations:
                existing_relations.append(
                    Relation(
                        label=relation[1],
                        source_id=relation[0],
                        target_id=relation[2],
                        properties={},
                    )
        )

        
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        

        print("node has been analyzed")
        return node
        

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        print(f"Number of nodes: {len(nodes)}")
        for node in nodes:
            jobs.append(self._aextract(node))
        
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )