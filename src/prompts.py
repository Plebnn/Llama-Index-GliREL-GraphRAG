from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

DEFAULT_ENTITIES = [
    # Core Entities
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "GPE",          # Geopolitical Entity (countries, cities, states)
    "PRODUCT",

    # Numerical & Temporal
    "DATE",
    "TIME",
    "MONEY",
    "QUANTITY",

    # Abstract & Cultural
    "EVENT",
]

DEFAULT_RELATIONS = {
    # Person to Person
    "spouse_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["PERSON"]
    },
    "child_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["PERSON"]
    },
    "parent_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["PERSON"]
    },
    "sibling_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["PERSON"]
    },
    
    # Person to Organization
    "member_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["ORGANIZATION"]
    },
    "founder_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["ORGANIZATION"]
    },
    "employee_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["ORGANIZATION"]
    },
    "CEO_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["ORGANIZATION"]
    },

    # Person to Location/GPE
    "lives_in": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["LOCATION", "GPE"]
    },
    "born_in": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["LOCATION", "GPE"]
    },
    "died_in": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["LOCATION", "GPE"]
    },
    "leader_of": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["GPE", "ORGANIZATION"]
    },
    "visited": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["LOCATION", "GPE"]
    },

    # Person to Temporal/Numerical
    "born_on": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["DATE"]
    },
    "died_on": {
        "allowed_head": ["PERSON"],
        "allowed_tail": ["DATE"]
    },

    # Organization to Other
    "headquartered_in": {
        "allowed_head": ["ORGANIZATION"],
        "allowed_tail": ["LOCATION", "GPE"]
    },
    "founded_on": {
        "allowed_head": ["ORGANIZATION"],
        "allowed_tail": ["DATE"]
    },
    "subsidiary_of": {
        "allowed_head": ["ORGANIZATION"],
        "allowed_tail": ["ORGANIZATION"]
    },
    "produces": {
        "allowed_head": ["ORGANIZATION"],
        "allowed_tail": ["PRODUCT"]
    },

    # Location/GPE Relations
    "capital_of": {
        "allowed_head": ["GPE", "LOCATION"],
        "allowed_tail": ["GPE"]
    },
    "located_in": {
        "allowed_head": ["LOCATION", "EVENT", "ORGANIZATION"],
        "allowed_tail": ["LOCATION", "GPE"]
    },

    # Event Relations
    "occurred_on": {
        "allowed_head": ["EVENT"],
        "allowed_tail": ["DATE"]
    },
    "occurred_at": {
        "allowed_head": ["EVENT"],
        "allowed_tail": ["TIME"]
    },
    "participant_in": {
        "allowed_head": ["PERSON", "ORGANIZATION"],
        "allowed_tail": ["EVENT"]
    },

    # Product Relations
    "costs": {
        "allowed_head": ["PRODUCT"],
        "allowed_tail": ["MONEY"]
    },
    "released_on": {
        "allowed_head": ["PRODUCT"],
        "allowed_tail": ["DATE"]
    }
}


DEFAULT_RECURSIVE_START_EXTRACT_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context. If there are no entity or relation types present choose your own.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: The 2023 wildfire season in Canada, which was likely exacerbated by climate change, began on May 1st, 2023. " 
    "By June 28th, 2023, the fires had burned over 8 million hectares of land. The smoke from these fires led to significant air quality degradation across North America,"
    " causing numerous health advisories to be issued. A state of emergency was declared in several provinces\n"
    "Output:\n"
    "[{{'head': 'climate change', 'head_type': 'CONCEPT', 'relation': 'CAUSED', 'tail': '2023 wildfire season in Canada', 'tail_type': 'EVENT'}},\n" # CASUAL RELATION
     "{{'head': '2023 wildfire season in Canada', 'head_type': 'EVENT', 'relation': 'STARTED_ON', 'tail': 'May 1st, 2023', 'tail_type': 'DATE'}},\n" # TEMPORAL RELATION
     "{{'head': 'May 1st, 2023', 'head_type': 'DATE', 'relation': 'PRECEDES', 'tail': 'June 28th, 2023', 'tail_type': 'DATE'}},\n" # TEMPORAL RELATION
     "{{'head': '2023 wildfire season in Canada', 'head_type': 'EVENT', 'relation': 'HAS_EFFECT', 'tail': 'burned over 8 million hectares of land', 'tail_type': 'QUANTITATIVE_MEASUREMENT'}},\n" # QUANTITIVE RELATION
     "{{'head': 'smoke from these fires', 'head_type': 'ENVIRONMENTAL_HAZARD', 'relation': 'CAUSED', 'tail': 'significant air quality degradation', 'tail_type': 'EFFECT'}},\n" # CASUAL RELATION
     "{{'head': 'significant air quality degradation', 'head_type': 'EFFECT', 'relation': 'LED_TO', 'tail': 'numerous health advisories', 'tail_type': 'ACTION'}},\n" # CROSS SENTENCE RELATION
     "{{'head': '2023 wildfire season in Canada', 'head_type': 'EVENT', 'relation': 'HAS_ARGUMENT', 'tail': 'state of emergency', 'tail_type': 'STATUS'}}]\n" #EVENT ARGUMENT RELATION
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: The European Union, a political and economic union, has 27 member states. Germany, one of the founding members, is a federal republic."
    "The German government is seated in Berlin. While membership in the EU requires adherence to certain democratic principles,"
    " it does not necessitate a specific form of government.\n"
    "Output:\n"
    "[{{'head': 'European Union', 'head_type': 'ORGANIZATION', 'relation': 'HAS_MEMBER_COUNT', 'tail': '27', 'tail_type': 'NUMBER'}},\n"
    "{{'head': 'Germany', 'head_type': 'COUNTRY', 'relation': 'IS_A', 'tail': 'founding member', 'tail_type': 'STATUS'}},\n"
    "{{'head': 'Germany', 'head_type': 'COUNTRY', 'relation': 'PART_OF', 'tail': 'European Union', 'tail_type': 'ORGANIZATION'}},\n" # HIRACHIAL RELATION
    "{{'head': 'Berlin', 'head_type': 'CITY', 'relation': 'LOCATION_OF', 'tail': 'German government', 'tail_type': 'GOVERNMENT'}},\n" # HIRACHIAL RELATION
    "{{'head': 'membership in the EU', 'head_type': 'CONCEPT', 'relation': 'POSSIBLY_REQUIRES', 'tail': 'adherence to certain democratic principles', 'tail_type': 'REQUIREMENT'}},\n" # MODAL RELATION
    "{{'head': 'membership in the EU', 'head_type': 'CONCEPT', 'relation': 'DOES_NOT_NECESSITATE', 'tail': 'a specific form of government', 'tail_type': 'CONDITION'}}]\n" # MODAL RELATION
    "---------------------\n"

    "Text: {text}\n"
    "Output:\n"
)

DEFAULT_RECURSIVE_START_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_RECURSIVE_START_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

DEFAULT_RECURSIVE_CHECK_TMPL = (
    "Check if the most important entities and relations in this text have been found.\n"
    "If the entities and relationships that have been found are the most important ones in the text and are enough to give an understanding about the Text, answer with: YES\n"
    "If there are important entities or relations left, answer with: NO\n"
    "Only answer with YES or NO\n"
    "---------------------\n"
    "Text: {text}\n"
    "Found entities: {existing_nodes}\n" 
    "Found relations: {existing_relations}\n"
    "Output (YES/NO):\n"

)

DEFAULT_RECURSIVE_CHECK_PROMPT = PromptTemplate(
    DEFAULT_RECURSIVE_CHECK_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

DEFAULT_RECURSIVE_LOOP_EXTRACT_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context. If there are no entity or relation types present choose your own.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "- Some relations have allready been found. Do NOT include these in your output!\n"
    "- Look also for unfound relationships of previously identified entities\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: The 2023 wildfire season in Canada, which was likely exacerbated by climate change, began on May 1st, 2023. " 
    "By June 28th, 2023, the fires had burned over 8 million hectares of land. The smoke from these fires led to significant air quality degradation across North America,"
    " causing numerous health advisories to be issued. A state of emergency was declared in several provinces\n"
    "\n"
    "Previously found entities: ['climate change','2023 wildfire season in Canada,'May 1st, 2023','significant air quality degradation','state of emergency','numerous health advisories']\n"
    "Previously found relations: [['2023 wildfire season in Canada','HAS_EFFECT','burned over 8 million hectares of land'],['significant air quality degradation','LED_TO',numerous health advisories'],['2023 wildfire season in Canada','STARTED_ON','May 1st, 2023']]\n"
    "Output:\n"
    "[{{'head': 'climate change', 'head_type': 'CONCEPT', 'relation': 'CAUSED', 'tail': '2023 wildfire season in Canada', 'tail_type': 'EVENT'}},\n" # CASUAL RELATION
     "{{'head': 'May 1st, 2023', 'head_type': 'DATE', 'relation': 'PRECEDES', 'tail': 'June 28th, 2023', 'tail_type': 'DATE'}},\n" # TEMPORAL RELATION
     "{{'head': 'smoke from these fires', 'head_type': 'ENVIRONMENTAL_HAZARD', 'relation': 'CAUSED', 'tail': 'significant air quality degradation', 'tail_type': 'EFFECT'}},\n" # CASUAL RELATION
     "{{'head': '2023 wildfire season in Canada', 'head_type': 'EVENT', 'relation': 'HAS_ARGUMENT', 'tail': 'state of emergency', 'tail_type': 'STATUS'}}]\n" #EVENT ARGUMENT RELATION
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: The European Union, a political and economic union, has 27 member states. Germany, one of the founding members, is a federal republic."
    "The German government is seated in Berlin. While membership in the EU requires adherence to certain democratic principles,"
    " it does not necessitate a specific form of government.\n"
    "\n"
    "Previously found entities: ['Germany','European Union,'membership in the EU','Berlin','German government','founding member']\n"
    "Previously found relations: [['Germany','IS_A','founding member'],['Berlin','LOCATION_OF','German government'],['membership in the EU','DOES_NOT_NECESSITATE','a specific form of government']]\n"
    "Output:\n"
    "[{{'head': 'European Union', 'head_type': 'ORGANIZATION', 'relation': 'HAS_MEMBER_COUNT', 'tail': '27', 'tail_type': 'NUMBER'}},\n"
    "{{'head': 'Germany', 'head_type': 'COUNTRY', 'relation': 'PART_OF', 'tail': 'European Union', 'tail_type': 'ORGANIZATION'}},\n" # HIRACHIAL RELATION
    "{{'head': 'Berlin', 'head_type': 'CITY', 'relation': 'CAPITAL_OF', 'tail': 'GERMANY', 'tail_type': 'GOVERNMENT'}},\n" # HIRACHIAL RELATION
    "{{'head': 'membership in the EU', 'head_type': 'CONCEPT', 'relation': 'POSSIBLY_REQUIRES', 'tail': 'adherence to certain democratic principles', 'tail_type': 'REQUIREMENT'}}]\n" # MODAL RELATION
     
    "---------------------\n"

    "Text: {text}\n"
    "\n"
    "Previously found entities: {existing_nodes}\n"
    "Previously found relations: {existing_relations}\n"
    "Output:\n"
)

DEFAULT_RECURSIVE_LOOP_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_RECURSIVE_LOOP_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

