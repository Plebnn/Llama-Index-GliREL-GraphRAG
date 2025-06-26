DEFAULT_ENTITIES = [
                "Person",
                "Organization",
                "Location",
                "Political Event",
                "Document/Publication",
                "Concept",
                "Award"
            ]
DEFAULT_RELATIONS = {
        "was President of": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Location"]
        },
        "was a member of": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Organization"]
        },
        "married": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Person"]
        },
        "defeated in election": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Person"]
        },
        "was Vice President under": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Person"]
        },
        "authored": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Document/Publication"]
        },
        "received": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Award"]
        },
        "visited": {
            "allowed_head": ["Person"],
            "allowed_tail": ["Location"]
        }
    }


