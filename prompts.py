from parser import args

rel_system_prompt = (
    "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
    "Following, you'll find a scientific TEXT, a desired OUTPUT FORMAT and a USER QUESTION. "
    "First, read the TEXT and study the OUTPUT FORMAT, then answer the USER QUESTION."
)

cot_models = ["llama33", "llama31"]
cot_prompt = (
    " Let's think step by step." if args.model in cot_models and args.force_cot else ""
)

judge_system_prompt = (
    "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
    "Following, you'll find a TEXT in the form of a scientific paper, "
    "a TRIPlE that specifies a molecular relationship, "
    "REASONING THOUGHTS denoting why this relationship has been extractd from the TEXT "
    " and a USER QUESTION. "
    "First, examine the TEXT, the TRIPLE and the REASONING THOUGHTS, then answer the USER QUESTION."
)

judge_prompt = (
    "The TRIPLE has been extracted by an AI, but an expert analysis shows "
    "that the relationship is factual wrong. Please analyse the TEXT with regards to the "
    "the REASONING THOUGHTS of the AI and explain very briefely why the reasoning process of the AI lead to the "
    "erroneous relationship. "
    "Use the given JSON OUTPUT FORMAT to summarize why the AI incorrectly inferred this relationship, e.g.: "
    '```json { "reason": "The AI incorrectly assumed... "```'
)

corrector_system_prompt = (
    "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
    "Following, you'll find a TEXT in the form of a scientific paper, "
    "a TRIPlE that specifies a molecular relationship, "
    "REASONING THOUGHTS denoting why this relationship has been extractd from the TEXT "
    " and a USER QUESTION. "
    "First, examine the TEXT, the TRIPLE and the REASONING THOUGHTS, then answer the USER QUESTION."
)

corrector_prompt = (
    "The TRIPLE has been extracted by an AI. You, as an expert analyst,  are tasked to re-evaluate the relationship "
    "and tell us if this triple is factually correct or incorrect. To do so, you will inspect the TEXT and the previous "
    "REASONING THOUGHTS of the AI and then output 'correct' or 'incorrect' dependent on your judgement. "
    "Use the given json OUTPUT FORMAT to output your answer."
)

chat_prompts = {
    "direct": {
        "oneshot": {
            "ppi": [
                "Extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
            "tf": [
                "Extract all the relations involving transcription factors to the target genes they regulate from the text. "
                "Please stick to the desired OUTPUT FORMAT ."
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
        },
        "stepwise": {
            "ppi": [
                "Extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Now review your extracted protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain only signalling pathway interactions "
                "and remove the rest. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Review one more time the protein-protein interactions (PPI's) to  "
                "determine whether there are in the list regulations that are of a transcriptional or gene  "
                "regulatory nature. Retain those interactions that are only specific to PPI's in cell  "
                "signalling and remove those relations that represent relations between transcription factors "
                "to their gene targets. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
            "tf": [
                "Extract all the relations involving transcription factors to the target genes they regulate from the text.",
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Now review your extracted transcription factor (TF) to gene relations to determine if "
                "they are specific to gene regulatory networks. Retain those interactions that are only "
                "involving TF's and their gene targets and remove those that are not. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Review one more time the transcription factor to gene relations  "
                "to determine whether there are in the list relations that are protein-protein "
                "interactions (PPI's) network or involved in protein signalling networks. Retain interactions  "
                "of gene regulatory networks involve a transcription factor and the gene whose expression  "
                "they regulate. Remove those relations that involve interactions between two signalling protein "
                "and PPI's. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
        },
    },
    "nerrel": {
        "oneshot": {
            "ppi": [
                "Extract all proteins involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the list above containing extracted proteins. Use it to extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
            "tf": [
                "Extract all transcription factors and genes from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the list above containing extracted transcription factors and genes. Use it to extract all the relations involving transcription factors to the target genes they regulate from the text. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
        },
        "stepwise": {
            "ppi": [
                "Extract all proteins involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the list above containing extracted proteins. Use it to extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Now review your extracted protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain only signalling pathway interactions "
                "and remove the rest. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Review one more time the protein-protein interactions (PPI's) to  "
                "determine whether there are in the list regulations that are of a transcriptional or gene  "
                "regulatory nature. Retain those interactions that are only specific to PPI's in cell  "
                "signalling and remove those relations that represent relations between transcription factors "
                "to their gene targets. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
            "tf": [
                "Extract all transcription factors and genes from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the list above containing extracted transcription factors and genes. Use it to extract all the relations involving transcription factors to the target genes they regulate from the text. ",
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Now review your extracted transcription factor (TF) to gene relations to determine if "
                "they are specific to gene regulatory networks. Retain those interactions that are only "
                "involving TF's and their gene targets and remove those that are not. ",
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
                "Review one more time the transcription factor to gene relations  "
                "to determine whether there are in the list relations that are protein-protein "
                "interactions (PPI's) network or involved in protein signalling networks. Retain interactions  "
                "of gene regulatory networks involve a transcription factor and the gene whose expression  "
                "they regulate. Remove those relations that involve interactions between two signalling protein "
                "and PPI's. "
                "Please stick to the desired OUTPUT FORMAT. "
                f"Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'.{cot_prompt}",
            ],
        },
    },
}

lookup = args.extractionmode if not args.all_ners_given else "nerrel"
prompts = chat_prompts[lookup][args.chattype][args.target]

OUTPUT_FORMAT = """
Use the following OUTPUT FORMAT:{
    // list of triples that describe interactions between two biological entities
    triples: [
    {
        // head entity of the triple 
        head: string,
        // relationship type
        relation: "INTERACTS_WITH",
        // tail entity name of the triple
        tail: string,
    }
    ],
}
"""
