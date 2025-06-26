from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from prompts import DEFAULT_ENTITIES, DEFAULT_RELATIONS
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode

from functools import cache
from typing import List, Tuple, Optional, Literal
import glirel  # noqa: F401 Import time side effect
import spacy


@cache
def nlp_model(threshold: float, entity_types: tuple[str], device: str):
    """Instantiate a spacy model with GLiNER and GLiREL components."""
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": entity_types,
        "style": "ent",
        "threshold": threshold,
        "map_location": device,
    }
    spacy.require_gpu()  # type: ignore

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    nlp.add_pipe("glirel", after="gliner_spacy")
    return nlp

#from tiny graphrag project
def extract_rels(
    text: str,
    device: str,
    entity_types: List[str],
    relation_types: List[str],
    threshold: float = 0.75
    ):
    """Extract entities and relations from text using GLiNER and GLiREL."""
    nlp = nlp_model(threshold, tuple(entity_types), device = device)
    docs = list(nlp.pipe([(text, {"glirel_labels":relation_types})], as_tuples=True))
    relations = docs[0][0]._.relations
    
    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)

    # Extract entities
    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]

    # Extract relations
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        #if item["score"] >= threshold           #threshold set for NER to high for rels
    ]
    
    unique_ents = set(ents)
    unique_rels = set(rels)

    return unique_ents, unique_rels



class GlirelPathExtractor(TransformComponent):
    
    entity_labels: List[str]
    relation_schema: dict
    device: str
    gliner_model_name: str
     
    def __init__(self, 
                entity_labels: Optional[List[str]] = None,
                relation_schema: Optional[dict] = None,
                device: Literal["cuda","mps","cpu"] = None,
                gliner_model_name: Optional[str] = None,
                #glirel_model_name,
                **kwargs: any
                 ):
        """Init params."""
        
        resolved_entities = entity_labels or DEFAULT_ENTITIES
        resolved_schema = relation_schema or DEFAULT_RELATIONS
        resolved_model_name = gliner_model_name or "urchade/gliner_mediumv2.1"

        super().__init__(
            entity_labels=resolved_entities,
            relation_schema=resolved_schema,
            device=device,
            gliner_model_name=resolved_model_name,
            **kwargs,  # Pass any extra arguments up the chain
        )
        
        


    def __call__(
        self, llama_nodes: list[BaseNode], **kwargs
    ) -> list[BaseNode]:
        for llama_node in llama_nodes:
            
            # extract existing relations from node
            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            # get text from nodes
            text = llama_node.get_content(metadata_mode=MetadataMode.LLM)
                        
            entities, relations = extract_rels(text=text, entity_types=self.entity_labels,relation_types=self.relation_schema, threshold=0.75, device=self.device)
            
            if entities:
                for entity in entities:
                    existing_nodes.append(
                        EntityNode(
                            name=entity[0], 
                            label=entity[1], 
                            properties={}   #GliREL does not extract values like description because it is not an llm, to include values like score is not nessecary because of the cut-off value
                        )
            )
            else:
                print("NO ENTITIES FOUND FOR THIS NODE!!!")

            if relations:                   # relation looks like this: ('Obama', 'received', 'Ripple of Hope Award')
                for relation in relations:
                    existing_relations.append(
                        Relation(
                            label=relation[1],
                            source_id=relation[0],
                            target_id=relation[2],
                            properties={},
                        )
            )
            

            # add back to the metadata

            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes

