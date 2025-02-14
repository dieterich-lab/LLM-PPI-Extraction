from parser import args

system_prompt = (
    "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
    "Following, you'll find a scientific TEXT, a desired OUTPUT FORMAT and a USER QUESTION. "
    "First, read the TEXT and study the OUTPUT FORMAT, then answer the USER QUESTION."
)

chat_prompts = {
    "direct": {
        "oneshot": {
            "ppi": [
                "Please extract all the protein-protein interactions involved in signalling pathways from the text."
            ],
            "tf": [
                "Please extract all the relations involving transcription factors to the target genes they regulate from the text."
            ],
        },
        "stepwise": {
            "ppi": [
                "Please extract all the protein-protein interactions involved in signalling pathways from the text.",
                "Now review your extracted protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain only signalling pathway interactions "
                "and remove the rest.",
                "Review one more time the protein-protein interactions (PPI's) to  "
                "determine whether there are in the list regulations that are of a transcriptional or gene  "
                "regulatory nature. Retain those interactions that are only specific to PPI's in cell  "
                "signalling and remove those relations that represent relations between transcription factors "
                "to their gene targets.",
            ],
            "tf": [
                "Please extract all the relations involving transcription factors to the target genes they regulate from the text.",
                "Now review your extracted transcription factor (TF) to gene relations to determine if "
                "they are specific to gene regulatory networks. Retain those interactions that are only "
                "involving TF's and their gene targets and remove those that are not.",
                "Review one more time the transcription factor to gene relations  "
                "to determine whether there are in the list relations that are protein-protein "
                "interactions (PPI's) network or involved in protein signalling networks. Retain interactions  "
                "of gene regulatory networks involve a transcription factor and the gene whose expression  "
                "they regulate. Remove those relations that involve interactions between two signalling protein "
                "and PPI's.",
            ],
        },
    },
    "nerrel": {
        "oneshot": {
            "ppi": [
                "Please extract all proteins involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the ENTITY LIST above with extracted proteins. Use it to extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
            ],
            "tf": [
                "Please extract all transcription factors and genes from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Look at the ENTITY LIST above with extracted transcription factors and genes. Use it to extract all the relations involving transcription factors to the target genes they regulate from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
            ],
        },
        "stepwise": {
            "ppi": [
                "Please extract all proteins involved in signalling pathways from the text.",
                "Look at the ENTITY LIST above with extracted proteins. Use it to extract all the protein-protein interactions involved in signalling pathways from the text. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Now review your extracted protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain only signalling pathway interactions "
                "and remove the rest. "
                "Please stick to the desired OUTPUT FORMAT.",
                "Review one more time the protein-protein interactions (PPI's) to  "
                "determine whether there are in the list regulations that are of a transcriptional or gene  "
                "regulatory nature. Retain those interactions that are only specific to PPI's in cell  "
                "signalling and remove those relations that represent relations between transcription factors "
                "to their gene targets. "
                "Please stick to the desired OUTPUT FORMAT.",
            ],
            "tf": [
                "Please extract all transcription factors and genes from the text.",
                "Look at the ENTITY LIST above with extracted transcription factors and genes. Use it to extract all the relations involving transcription factors to the target genes they regulate from the text. ",
                "Please stick to the desired OUTPUT FORMAT.",
                "Now review your extracted transcription factor (TF) to gene relations to determine if "
                "they are specific to gene regulatory networks. Retain those interactions that are only "
                "involving TF's and their gene targets and remove those that are not. ",
                "Please stick to the desired OUTPUT FORMAT.",
                "Review one more time the transcription factor to gene relations  "
                "to determine whether there are in the list relations that are protein-protein "
                "interactions (PPI's) network or involved in protein signalling networks. Retain interactions  "
                "of gene regulatory networks involve a transcription factor and the gene whose expression  "
                "they regulate. Remove those relations that involve interactions between two signalling protein "
                "and PPI's. "
                "Please stick to the desired OUTPUT FORMAT.",
            ],
        },
    },
}

prompts = chat_prompts[args.extractionmode][args.chattype][args.target]
