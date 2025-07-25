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

        # Get existing data from metadata
        existing_nodes_objects = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations_objects = node.metadata.pop(KG_RELATIONS_KEY, [])
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        
        # Use sets to store unique raw triplets and prevent duplicates
        unique_entities = {(e.name, e.label) for e in existing_nodes_objects}
        unique_relations = {(r.source_id, r.label, r.target_id) for r in existing_relations_objects}

        # Initial extraction if no data exists
        if not existing_nodes_objects and not existing_relations_objects:
            new_entities, new_relations = await self._start_extraction(
                text=text,
                prompt=self.extract_prompt
            )
            unique_entities.update(new_entities)
            unique_relations.update(new_relations)

        # Recursive checking and extraction loop
        count = 0
        while count < self.max_loops:
            # Check if the LLM thinks it's done
            is_complete = await self._check_if_complete(
                text=text, 
                prompt=DEFAULT_RECURSIVE_CHECK_PROMPT,
                existing_nodes=str(list(unique_entities)), 
                existing_relations=str(list(unique_relations))
            )
            print(f"Is extraction complete? {is_complete}")
            if is_complete or len(unique_relations) >= self.max_paths_per_chunk:
                if len(unique_relations) >= self.max_paths_per_chunk:
                    print("Max number of relations per chunk found.")
                break

            count += 1
            print(f"Starting extraction loop #{count}")
            
            new_entities, new_relations = await self._loop_extraction(
                text=text,
                existing_nodes=str(list(unique_entities)),
                existing_relations=str(list(unique_relations)),
                prompt=DEFAULT_RECURSIVE_LOOP_EXTRACT_PROMPT
            )
            unique_entities.update(new_entities)
            unique_relations.update(new_relations)
        
        # --- START: CORRECTED LOGIC ---

        # 1. Create all EntityNode objects and store them in a dict for easy lookup.
        #    The key is the entity name, the value is the EntityNode object.
        final_nodes_map = {
            name: EntityNode(name=name, label=label) for name, label in unique_entities
        }

        # 2. Create Relation objects using the .id from the nodes in the map.
        final_relations = []
        for head_name, rel_label, tail_name in unique_relations:
            # Ensure both head and tail nodes exist in our map before creating a relation
            if head_name in final_nodes_map and tail_name in final_nodes_map:
                source_node = final_nodes_map[head_name]
                target_node = final_nodes_map[tail_name]
                final_relations.append(
                    Relation(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        label=rel_label,
                    )
                )

        final_nodes = list(final_nodes_map.values())
        
        if not final_nodes:
            print("NO ENTITIES FOUND FOR THIS NODE!!!")

        # --- END: CORRECTED LOGIC ---

        # Update node metadata
        node.metadata[KG_NODES_KEY] = final_nodes
        node.metadata[KG_RELATIONS_KEY] = final_relations

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