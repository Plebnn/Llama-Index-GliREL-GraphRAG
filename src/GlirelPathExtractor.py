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
    ents = [
        (" ".join(ent.text.split()), ent.label_) for ent in docs[0][0].ents
    ]

    # Extract relations
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= 0.4          #threshold set for NER to high for rels
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
            # get text from node
            text = llama_node.get_content(metadata_mode=MetadataMode.LLM)
            
            # Extract raw entities and relations using your function
            entities, relations = extract_rels(
                text=text,
                entity_types=self.entity_labels,
                relation_types=self.relation_schema,
                threshold=0.75,
                device=self.device
            )
            
            if not entities:
                print("NO ENTITIES FOUND FOR THIS NODE!!!")
                # If there are no entities, we can't have relations, so we can skip the rest
                llama_node.metadata[KG_NODES_KEY] = []
                llama_node.metadata[KG_RELATIONS_KEY] = []
                continue

            # --- START: CORRECTED LOGIC ---

            # 1. Create a dictionary to map entity names to EntityNode objects for quick lookup.
            #    This also handles deduplication of nodes.
            final_nodes_map = {
                name: EntityNode(name=name, label=label) 
                for name, label in entities
            }

            # 2. Create relations by looking up the nodes in the map and using their .id
            final_relations = []
            if relations:
                for head_name, rel_label, tail_name in relations:
                    # Ensure both the source and target entities were extracted before creating a relation
                    if head_name in final_nodes_map and tail_name in final_nodes_map:
                        source_node = final_nodes_map[head_name]
                        target_node = final_nodes_map[tail_name]
                        final_relations.append(
                            Relation(
                                label=rel_label,
                                source_id=source_node.id, # Use the unique ID
                                target_id=target_node.id, # Use the unique ID
                                properties={},
                            )
                        )
            
            # --- END: CORRECTED LOGIC ---

            # 3. Add the final lists of nodes and relations back to the metadata
            llama_node.metadata[KG_NODES_KEY] = list(final_nodes_map.values())
            llama_node.metadata[KG_RELATIONS_KEY] = final_relations

        return llama_nodes

