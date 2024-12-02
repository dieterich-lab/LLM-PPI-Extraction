INTERACT_TEMPLATE = """Based on the following example, extract entities and 
relations from the provided text.

Use the following entity types, don't use other entity that is not defined below:
# ENTITY TYPES:
{node_labels}

Use the following relation types, don't use any other relation that is not defined below:
# RELATION TYPES:
{rel_types}

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text, extract entities and relations as in the provided example:

{format_instructions}

Text: {input}

"""

INTERACT_TEMPLATE_SIMPLE = """Based on the following example, extract entities and 
relations from the provided text.

Use the following relation types, don't use any other relation that is not defined below:
# RELATION TYPES:
{rel_types}

Below are a number of examples of text and their extracted entities and relationships.
{examples}

Use the following JSON format:

{format_instructions}

### Text: 

{input}

"""

NER_TEMPLATE_SIMPLE = """Based on the following example, extract all proteins
from the provided text.

Below are a number of examples of text passages and corresponding extracted proteins.
{examples}

Use the following JSON format:

{format_instructions}

### Text: 
 
{input}

"""

PPI_BASESTRINGPARTS = [
    "You are a top-tier molecular biologist specialized in the field of cardiology. "
    "Your task is to identify pairs of proteins which are known to be interacting with "
    " each other. It is very important that the entities that you identify as interacting "
    "to only be proteins. It is import that you don't confuse them with transcription factors. "
    "To me, it does not matter the nature of the interaction, it can be "
    "an activation, inhibition, binding, etc.. All that matters is that you provide to me "
    "pairs of proteins which are known to be interacting with each other one way or another."
    "the protein (entities) and relations (the interaction between the proteins) "
    " requested with the user prompt from a given "
    "text. You must generate the output in a JSON format containing a list "
    "with JSON objects. "
    'Each object should have the keys: "head", '
    '"relation" and "tail". The "head" and "tail" '
    "key must contain the name or denominator of the extracted protein."
    "Attempt to extract as many proteins and relations as you can. Maintain "
    "Entity Consistency: When extracting entities, it's vital to ensure "
    'consistency. If a protein, such as "PRKACA", is mentioned multiple '
    "times in the text but is referred to by different names "
    '(e.g., "PRKACA", "PKACA", "cAMP-activated catalytic subunit alpha"), '
    "always use the canonical gene name identifier for "
    "that entity. The knowledge graph should be coherent and easily "
    "understandable, so maintaining consistency in entity references is "
    "crucial.",
    "IMPORTANT NOTES: Don't add any explanation and text.",
]

PPI_EXAMPLES = [
    {
        "text": ("BNIP-2 Interacts with LATS1 to Promote YAP Cytosolic Localization"),
        "head": "BNIP-2",
        "head_type": "Protein",
        "relation": "INTERACTS_WITH",
        "tail": "LATS1",
        "tail_type": "Protein",
    },
    {
        "text": (
            "CBY1 interacts with DZIP1 and "
            "localizes to the basal body in developing mitral valves."
        ),
        "head": "CBY1",
        "head_type": "Protein",
        "relation": "INTERACTS_WITH",
        "tail": "DZIP1",
        "tail_type": "Protein",
    },
    {
        "text": (
            "CAMK2 kinase induces cardiac hypertrophy and "
            "activates MEF2 transcription factor in vivo."
        ),
        "head": "CAMK2",
        "head_type": "Protein",
        "relation": "INTERACTS_WITH",
        "tail": "MEF2",
        "tail_type": "Protein",
    },
    {
        "text": "The reduced 14-3-3 co-immunoprecipitation experiments suggest that PKA inhibits HDAC4 activity.",
        "head": "PKA",
        "head_type": "Protein",
        "relation": "INTERACTS_WITH",
        "tail": "HDAC4",
        "tail_type": "Protein",
    },
    {
        "text": "TEL2 binds to TTI1 and both TEL2 and TTI1 are necessary and sufficient to stabilize and activate both mTORC1 and mTORC2 signalling pathways.",
        "head": "TEL2",
        "head_type": "Protein",
        "relation": "INTERACTS_WITH",
        "tail": "TTI1",
        "tail_type": "Protein",
    },
]

TF_BASESTRINGPARTS = [
    "You are a top-tier molecular biologist specialized in the field of cardiology. "
    "Your task is to identify pairs of regulatory interactions between transcription factors and their target genes."
    "It is very important that the entities that you identify as in such relation pairs "
    "to only be transcription factors as a source and a gene as a target. "
    "It is important that you don't confuse them protein-protein interactions. "
    "To me, it does not matter the nature of the interaction, it can be "
    "either suppressive or enhancing. All that matters is that you identify as many transcription factors "
    " together with the genes that they are targeting."
    "the entities (transcription factors and their target genes) "
    " and relations (transcriptional relations)  requested with the user prompt from a given "
    "text. You must generate the output in a JSON format containing a list "
    "with JSON objects. "
    'Each object should have the keys: "head", '
    '"relation" and "tail". The "head" and "tail" '
    "key must contain the name or denominator of the extracted transcription factors and genes."
    "Again, attempt to extract as many proteins and relations as you can and provide the pairs with their gene names."
    "Also what is very important is that the names that you provide for the transcription factors and their target genes, "
    "should either be canonical gene names or their gene synonyms as they would be reported in ensembl."
    "For example if you are reporting a gene relation involving the Hypoxia-inducible factor 1-alpha transcription factor, "
    "this should either be reported by it's gene name HIF1A or as some other gene synonym from Ensembl, such as HIF1, HIF1-α, etc.."
    "The same convention should be also for the names of the target genes."
    "The knowledge graph should be coherent and easy "
    "understandable, so maintaining consistency in entity references is "
    "crucial.",
    "IMPORTANT NOTES: Don't add any explanation and text.",
]

PPI_BASESTRINGPARTS_SIMPLE = [
    "You are a top-tier molecular biologist specialized in the field of cardiology. "
    "Your task is to identify pairs of proteins which are known to be interacting with "
    " each other. "
    "You must generate the output in a JSON format containing a list with JSON objects. "
    'Each object should have the keys: "head", "relation" and "tail". '
    'The "head" and "tail" key must contain the name or denominator of the extracted  proteins / genes. '
    "The objects should be coherent and easy understandable, so maintaining consistency in entity references is "
    "crucial.",
    "IMPORTANT NOTES: Only extract objects for entities that appear in the input and ",
    "don't add any explanation!",
]

TF_BASESTRINGPARTS_SIMPLE = [
    "You are a top-tier molecular biologist specialized in the field of cardiology. "
    "Your task is to identify pairs of regulatory interactions between transcription factors and their target genes."
    "You must generate the output in a JSON format containing a list with JSON objects. "
    'Each object should have the keys: "head", "relation" and "tail". '
    'The "head" and "tail" key must contain the name or denominator of the extracted transcription factors and genes. '
    "The objects should be coherent and easy understandable, so maintaining consistency in entity references is "
    "crucial.",
    "IMPORTANT NOTES: Only extract objects for entities that appear in the input and ",
    "don't add any explanation!",
]

PPI_EXAMPLES_SIMPLE = [
    {
        "text": ("BNIP-2 Interacts with LATS1 to Promote YAP Cytosolic Localization"),
        "head": "BNIP-2",
        "relation": "INTERACTS_WITH",
        "tail": "LATS1",
    },
    {
        "text": (
            "CBY1 interacts with DZIP1 and "
            "localizes to the basal body in developing mitral valves."
        ),
        "head": "CBY1",
        "relation": "INTERACTS_WITH",
        "tail": "DZIP1",
    },
    {
        "text": (
            "CAMK2 kinase induces cardiac hypertrophy and "
            "activates MEF2 transcription factor in vivo."
        ),
        "head": "CAMK2",
        "relation": "INTERACTS_WITH",
        "tail": "MEF2",
    },
    {
        "text": "The reduced 14-3-3 co-immunoprecipitation experiments suggest that PKA inhibits HDAC4 activity.",
        "head": "PKA",
        "relation": "INTERACTS_WITH",
        "tail": "HDAC4",
    },
    {
        "text": "TEL2 binds to TTI1 and both TEL2 and TTI1 are necessary and sufficient to stabilize and activate both mTORC1 and mTORC2 signalling pathways.",
        "head": "TEL2",
        "relation": "INTERACTS_WITH",
        "tail": "TTI1",
    },
]

TF_EXAMPLES = [
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "head": "MEF2A",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "ZEB2",
        "tail_type": "gene",
    },
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "head": "MEF2A",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "CTNNB1",
        "tail_type": "gene",
    },
    {
        "text": (
            "CREM regulate the circadian expression of CYP51 and "
            "other cholesterogenic genes in the human heart."
        ),
        "head": "CREM",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "CYP51",
        "tail_type": "gene",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "IL-1",
        "tail_type": "gene",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "IL-6",
        "tail_type": "gene",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "IL-12",
        "tail_type": "gene",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "head_type": "transcription_factor",
        "relation": "REGULATES",
        "tail": "TNF-α",
        "tail_type": "gene",
    },
]
PPI_BASESTRINGPARTS_SIMPLE = [
    "You are a top-tier molecular biologist specialized in the field of cardiology. "
    "Your task is to identify pairs of proteins which are known to be interacting with "
    " each other. "
    "You must generate the output in a JSON format containing a list with JSON objects. "
    'Each object should have the keys: "head", "relation" and "tail". '
    'The "head" and "tail" key must contain the name or denominator of the extracted  proteins / genes. '
    "The objects should be coherent and easy understandable, so maintaining consistency in entity references is "
    "crucial.",
    "IMPORTANT NOTES: Only extract objects for entities that appear in the input and ",
    "don't add any explanation!",
]


TF_EXAMPLES_SIMPLE = [
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "head": "MEF2A",
        "relation": "REGULATES",
        "tail": "ZEB2",
    },
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "head": "MEF2A",
        "relation": "REGULATES",
        "tail": "CTNNB1",
    },
    {
        "text": (
            "CREM regulate the circadian expression of CYP51 and "
            "other cholesterogenic genes in the human heart."
        ),
        "head": "CREM",
        "relation": "REGULATES",
        "tail": "CYP51",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "relation": "REGULATES",
        "tail": "IL-1",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "relation": "REGULATES",
        "tail": "IL-6",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "relation": "REGULATES",
        "tail": "IL-12",
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "STAT3",
        "relation": "REGULATES",
        "tail": "TNF-α",
    },
]

LR_EXAMPLES_SIMPLE = [
    {
        "text": (
            "We have applied the multiscale framework to the interaction between ligand TNFα and its receptor TNFR1 as a test model."
        ),
        "head": "TNFα",
        "relation": "INTERACTS_WITH",
        "tail": "TNFR1",
    },
    {
        "text": (
            "INS binds to the insulin receptor (IR) on the plasma membrane (PM) and triggers the activation of signaling cascades to regulate metabolism and cell growth."
        ),
        "head": "INS",
        "relation": "INTERACTS_WITH",
        "tail": "IR",
    },
    {
        "text": (
            "The binding of the epidermal growth factor (EGF) to its receptor (EGFR) triggers a large set of downstream processes, ultimately causing cell growth, differentiation and proliferation"
        ),
        "head": "EGF",
        "relation": "INTERACTS_WITH",
        "tail": "EGFR",
    },
    {
        "text": (
            "TGFBR1 is the key component in passing extracellular stimulation to the downstream TGF-β signaling pathway."
        ),
        "head": "TGF-β",
        "relation": "INTERACTS_WITH",
        "tail": "TGFBR1",
    },
    {
        "text": (
            "TGFBR2 is the receptor that TGF-β binds directly, and thus it serves as a gatekeeper for the activation of downstream signaling."
        ),
        "head": "TGF-β",
        "relation": "INTERACTS_WITH",
        "tail": "TGFBR2",
    },
    {
        "text": (
            "PDGFRA can bind to and dimerize the PDGF ligands"
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "head": "PDGF",
        "relation": "INTERACTS_WITH",
        "tail": "PDGFRA",
    },
    {
        "text": (
            "Gq-dependent pathway- Ang II as a ligand (L) binds to AT1R (GPCR), activates Gq protein subunits which recruit the GRKs to the receptor."
        ),
        "head": "Ang II",
        "relation": "INTERACTS_WITH",
        "tail": "AT1R",
    },
]

PPI_NODE_LABELS = ["protein"]
LR_NODE_LABELS = ["ligand", "receptor"]
PPI_INTERACTIONS = ["INTERACTS_WITH"]
LR_INTERACTIONS = ["INTERACTS_WITH"]
TF_NODE_LABELS = ["transcription_factor", "gene"]
TF_INTERACTIONS = ["REGULATES"]

PPI_NER_EXAMPLES_SIMPLE = [
    {
        "text": ("BNIP-2 Interacts with LATS1 to Promote YAP Cytosolic Localization"),
        "proteins": ["BNIP-2", "LATS1"],
    },
    {
        "text": (
            "CBY1 interacts with DZIP1 and "
            "localizes to the basal body in developing mitral valves."
        ),
        "proteins": ["CBY1", "DZIP1"],
    },
    {
        "text": (
            "CAMK2 kinase induces cardiac hypertrophy and "
            "activates MEF2 transcription factor in vivo."
        ),
        "proteins": ["CAMK2", "MEF2"],
    },
    {
        "text": "The reduced 14-3-3 co-immunoprecipitation experiments suggest that PKA inhibits HDAC4 activity.",
        "proteins": ["PKA", "HDAC4"],
    },
    {
        "text": "TEL2 binds to TTI1 and both TEL2 and TTI1 are necessary and sufficient to stabilize and activate both mTORC1 and mTORC2 signalling pathways.",
        "proteins": ["TEL2", "TTI1"],
    },
]

TF_NER_EXAMPLES_SIMPLE = [
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "protiens": ["MEF2A", "ZEB2"],
    },
    {
        "text": (
            "MEF2A transcriptionally upregulates the expression of ZEB2 and CTNNB1"
        ),
        "proteins": ["MEF2A", "CTNNB1"],
    },
    {
        "text": (
            "CREM regulate the circadian expression of CYP51 and "
            "other cholesterogenic genes in the human heart."
        ),
        "proteins": ["CREM", "CYP51"],
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "proteins": ["STAT3", "IL-1"],
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "proteins": ["STAT3", "IL-6"],
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "proteins": ["STAT3", "IL-12"],
    },
    {
        "text": (
            "STAT3 then travels to the nucleus where it stimulates the transcription of specific genes, "
            "which in-turn are thought to abrogate the inflammatory response by transcriptionally repressing "
            "proinflammatory cytokine genes such as IL-1, IL-6, IL-12, and TNF-α."
        ),
        "proteins": ["STAT3", "TNF-α"],
    },
]
