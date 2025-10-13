from parser import args

rel_system_prompt = (
    "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
    "Following, you'll find a scientific TEXT, a desired OUTPUT FORMAT and a USER QUESTION. "
    "First, read the TEXT and study the OUTPUT FORMAT, then answer the USER QUESTION."
)

cot_models = ["llama33", "llama31", "llama33regu", "llama31regu"]


cot_prompt = (
    "Let's think step by step. " if args.model in cot_models and args.force_cot else ""
)
confidence_prompt = (
    "Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'."
    if not args.noconfidence
    else ""
)

# judge_system_prompt = (
#     "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
#     "Following, you'll find a TEXT in the form of a scientific paper, "
#     "a TRIPlE that specifies a molecular relationship, "
#     "REASONING THOUGHTS denoting why this relationship has been extractd from the TEXT "
#     " and a USER QUESTION. "
#     "First, examine the TEXT, the TRIPLE and the REASONING THOUGHTS, then answer the USER QUESTION."
# )

# judge_prompt = (
#     "The TRIPLE has been extracted by an AI, but an expert analysis shows "
#     "that the relationship is factual wrong. Please analyse the TEXT with regards to the "
#     "the REASONING THOUGHTS of the AI and explain very briefely why the reasoning process of the AI lead to the "
#     "erroneous relationship. "
#     "Use the given JSON OUTPUT FORMAT to summarize why the AI incorrectly inferred this relationship, e.g.: "
#     '```json { "reason": "The AI incorrectly assumed... "```'
# )

# corrector_system_prompt = (
#     "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
#     "Following, you'll find a TEXT in the form of a scientific paper, "
#     "a TRIPlE that specifies a molecular relationship, "
#     "REASONING THOUGHTS denoting why this relationship has been extractd from the TEXT "
#     " and a USER QUESTION. "
#     "First, examine the TEXT, the TRIPLE and the REASONING THOUGHTS, then answer the USER QUESTION."
# )

# corrector_prompt = (
#     "The TRIPLE has been extracted by an AI. You, as an expert analyst,  are tasked to re-evaluate the relationship "
#     "and tell us if this triple is factually correct or incorrect. To do so, you will inspect the TEXT and the previous "
#     "REASONING THOUGHTS of the AI and then output 'correct' or 'incorrect' dependent on your judgement. "
#     "Use the given json OUTPUT FORMAT to output your answer."
# )

targets = "proteins" if args.target == "ppi" else "transcription factors and genes"
anti_targets = "proteins" if args.target == "tf" else "transcription factors and genes"
target = (
    "protein"
    if args.target == "ppi"
    else (
        "transcription factor/gene"
        if args.target == "tf"
        else "protein and transcription factor/gene"
    )
)
interactions_type = (
    "protein-protein"
    if args.target == "ppi"
    else (
        "transcription factor-to-gene"
        if args.target == "tf"
        else "protein-protein and transcription factor-to-gene"
    )
)
anti_interactions_type = (
    "transcription factor/gene" if args.target == "ppi" else "protein-protein"
)

if args.extractionmode == "nerrel":
    ner_list_prompt = (
        f"Now look at your extracted {targets} above and use it for the following task:"
    )
elif args.all_ners_given:
    ner_list_prompt = f"Look at the list above. These are the ground truth {targets} that are found in the abstract. But be wary, not all of them will neccessarily be a particpant in {interactions_type} relations. Use this list for the following task:"
elif args.true_ners_given:
    ner_list_prompt = f"Look at the list above. These are the ground truth {targets} that are found in the abstract and also take part in {interactions_type} relations. Use this list for the following task:"
else:
    ner_list_prompt = ""

lookup_prompt = (
    "We also provided above some insightful BACKGROUND KNOWLEDGE for each extracted protein. Use it as additional support. "
    if args.lookup
    else ""
)

dynex_prompt = (
    "Following, you find an EXAMPLE of a similar texts and ground truth relations. Use it as support for your decision. "
    if args.dynex_k > 0
    else ""
)

if not args.recall:
    prompt = f"{ner_list_prompt} Extract all the {interactions_type} interactions involved in signalling pathways from the text. Please only extract {target} pairs which directly interact with each other (i.e. through binding, phosphorylation, sumoylation, etc). Do not misinterpret functional relationships, co-occurrence, structural similarity, or indirect regulatory effects for direct interactions. {lookup_prompt}{dynex_prompt}"
else:
    prompt = f"{ner_list_prompt} Extract ALL the relations between molecular entities from the text. Be as greedy as possible, we will filter the relations for correctness later in a second step {lookup_prompt}"

ppi_pos_ex = """
Below you find some positive examples between protein-protein relations that give you an idea of what we are looking for:
[
    {
        'text': (
            'This cytokine induces the p53 into a mutant-like conformation that forms a complex with Sp1' 
        ),
        'head': 'p53',
        'relation': 'INTERACTS_WITH',
        'tail': 'Sp1',
    },
    {
        'text': (
            'These findings suggest that the STAT3-NRF2 complex accelerates BLBC growth and progression by augmenting IL-23A expression.' 
        ),
        'head': 'STAT3',
        'relation': 'INTERACTS_WITH',
        'tail': 'NRF2',
    },
    {
        'text': (
            'HIF1A forms a transcriptional complex with ARNT under hypoxia.' 
        ),
        'head': 'HIF1A',
        'relation': 'INTERACTS_WITH',
        'tail': 'ARNT',
    },
    {
        'text': (
            'PRMT1 methylates cGAS and suppresses cGAS/STING signaling in cancer cells' 
        ),
        'head': 'PRMT1',
        'relation': 'INTERACTS_WITH',
        'tail': 'cGAS',
    },
    {
        'text': (
            'TRAF6 ubiquitinates TGFβ type I receptor to promote its cleavage and nuclear translocation in cancer.' 
        ),
        'head': 'TRAF6',
        'relation': 'INTERACTS_WITH',
        'tail': 'TGFβ',
    },
    {
        'text': 'AKT1 phosphorylates AKT1S1 at Thr-246.',
        'head': 'AKT1',
        'relation': 'INTERACTS_WITH',
        'tail': 'AKT1S1',
    },
    {
        'text': 'PIAS1 sumoylates PNKP in cells.',
        'head': 'PIAS1',
        'relation': 'INTERACTS_WITH',
        'tail': 'PNKP',
    },
{
        'text': 'CBP, but not p/CAF, acetylates GATA-1 at two highly conserved lysine-rich motifs present at the C-terminal tails of both zinc fingers.',
        'head': 'CBP',
        'relation': 'INTERACTS_WITH',
        'tail': 'GATA-1',
    },
]
"""

ppi_neg_ex = """
Below you find some examples of false positive protein-protein relations and the reason why you should not extract those:
* 'KRAS and BRAF cooperate in the MAPK signaling cascade to promote cell proliferation.': Although the two proteins are in the same signalling system, the text does not provide evidence of a direct interaction.
* 'p53 and Protein MYC are both found in the same signaling complex.': Incorrect assumptions based on co-occurrence or proximity.
* 'TNF and IL6 accumulate at DNA damage sites.': Co-localization but no evidence of direct relation/interaction between the two.
* 'Gene TNF regulates the expression of Gene IL6': Misinterpretation of genetic or signalling pathways as protein interactions.
* 'Prmt5 shares 80% sequence identity with Protein Prmt7, which is known to bind BRAF.': Incorrect assumptions based on structural similarity.
* 'PTEN was pulled down in a co-IP assay with CDKN2A.': Incorrect interpretations of experimental methods.
"""

tf_pos_ex = """
Below you find some positive examples of relations between transcription factor-to-gene relations that give you an idea of what we are looking for:
* "MYC target genes that are involved in cell cycle such as Cyclin D1": MYC -> Cyclin D1
* "STAT3 can induce the expression of anti-apoptotic genes like Bcl-2, which help in cell survival.": STAT3 -> Bcl-2
* "Tbx1 activates transcription of the fibroblast growth factor genes Fgf8 and Fgf10 to maintain proliferative expansion and inhibit differentiation of cardiopharyngeal precursor cells": Tbx1 -> Fgf8 and Tbx1 -> Fgf10
* "Overexpression of MECP2 leads to the suppression of IFN-γ transcription, which is linked to impaired TH1 responses in both children and mice with MECP2 duplication syndrome.": MECP2 -> IFN-γ
* "In humans, FOXO regulates the expression of core small RNA pathway genes, including AGO2.": FOXO -> AGO2.
"""

tf_neg_ex = """
Below you find some false positive examples of relations between transcription factor-to-gene relations and the reason why you should not extract those:
* "This cytokine induces the p53 into a mutant-like conformation that forms a complex with Sp1": This is complex formation and does not involve transcription factor to gene relations.
* "HIF1A forms a transcriptional complex with ARNT under hypoxia.": This relation represents two transcription factor proteins interacting with each other, and the text does not reflect that they target the regulation of any specific gene.
* "PRMT1 methylates cGAS and suppresses cGAS/STING signaling in cancer cells": This is a methylation interaction and not a transcription factor to gene relation.
* "Gene MYC and gene STAT3 share a common promoter region.": This is not explicitly a relation between a transcription factor and its target gene.
* "AKT1 phosphorylates AKT1S1 at Thr-246.". This is a phosphorylation interaction and not a transcription factor to gene relation.
"""

nn = "\n"
if args.target == "ppi":
    ex = f"{ppi_pos_ex if args.examples in ['negpos', 'pos'] else ''}{ppi_neg_ex if args.examples in ['negpos', 'neg'] else ''}"
elif args.target == "tf":
    ex = f"{tf_pos_ex if args.examples in ['negpos', 'pos'] else ''}{tf_neg_ex if args.examples in ['negpos', 'neg'] else ''}"
elif args.target == "ppitf":
    ex = f"{ppi_pos_ex + nn + tf_pos_ex if args.examples in ['negpos', 'pos'] else ''}{ppi_neg_ex + nn + tf_neg_ex if args.examples in ['negpos', 'neg'] else ''}"

chat_prompts = {
    "direct": {
        "oneshot": [
            f"{prompt} "
            f"{ex} "
            "Please stick to the desired OUTPUT FORMAT. "
            f"{confidence_prompt}{cot_prompt}",
        ],
        "stepwise": [
            f"{prompt} "
            f"{ex} "
            "Please stick to the desired OUTPUT FORMAT. "
            f"{confidence_prompt}{cot_prompt}",
            f"Now review your extracted {interactions_type} interactions to determine if "
            "they are specific to signaling pathways. Retain only signalling pathway interactions "
            "and remove the rest. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
            f"Review one more time the {interactions_type} interactions to  "
            f"determine whether there are in the list regulations that are of a {anti_interactions_type} "
            f"regulatory nature. Retain those interactions that are only specific to {interactions_type} interactions in cell  "
            f"signalling and remove those relations that represent relations between {anti_targets}. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
        ],
    },
    "nerrel": {
        "oneshot": [
            f"Extract all the {targets} that appear in the text. "
            "Please stick to the desired OUTPUT FORMAT. ",
            f"{prompt} "
            f"{ex} "
            "Please stick to the desired OUTPUT FORMAT. "
            f"{confidence_prompt}{cot_prompt}",
        ],
        "stepwise": [
            f"Extract all the {targets} that appear in the text.",
            f"{prompt} "
            f"{ex} "
            "Please stick to the desired OUTPUT FORMAT. "
            f"{confidence_prompt}{cot_prompt}",
            f"Now review your extracted {interactions_type} interactions to determine if "
            "they are specific to signaling pathways. Retain only signalling pathway interactions "
            "and remove the rest. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
            f"Review one more time the {interactions_type} interactions to  "
            f"determine whether there are in the list regulations that are of a {anti_interactions_type} "
            f"regulatory nature. Retain those interactions that are only specific to {interactions_type} interactions in cell  "
            f"signalling and remove those relations that represent relations between {anti_targets}. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
        ],
        "lookup": [  # Same as stepwise but with background knowledge
            f"Extract all the {targets} that appear in the text.",
            f"{prompt} "
            f"{ex} "
            "Please stick to the desired OUTPUT FORMAT. "
            f"{confidence_prompt}{cot_prompt}",
            f"Now review your extracted {interactions_type} interactions to determine if "
            "they are specific to signaling pathways. Retain only signalling pathway interactions "
            "and remove the rest. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
            f"Review one more time the {interactions_type} interactions to  "
            f"determine whether there are in the list regulations that are of a {anti_interactions_type} "
            f"regulatory nature. Retain those interactions that are only specific to {interactions_type} interactions in cell  "
            f"signalling and remove those relations that represent relations between {anti_targets}. "
            "Use again the desired json OUTPUT FORMAT to format your answer. "
            f"{confidence_prompt}{cot_prompt}",
        ],
    },
}

mode_lookup = (
    args.extractionmode
    if not (args.all_ners_given or args.true_ners_given)
    else "nerrel"
)
# chat_lookup = "stepwise" if args.chattype == "lookup" else args.chattype
prompts = chat_prompts[mode_lookup][args.chattype]
if args.target == "ppitf" and args.chattype == "stepwise":
    prompts = prompts[:-1]


OUTPUT_FORMAT = """
{
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

# ============================================================================
# Tree-of-Thoughts (ToT) Prompts
# ============================================================================

tot_strategy_generation_prompt = """
You are designing extraction strategies for identifying {interactions_type} interactions from scientific text.

Generate {n_paths} different reasoning approaches/strategies for extracting these relations. Each strategy should focus on a different aspect:
- Strategy 1: Focus on explicit interaction verbs (binds, phosphorylates, activates, etc.)
- Strategy 2: Focus on experimental evidence (co-IP, pull-down, reporter assays, etc.)
- Strategy 3: Focus on functional descriptions and mechanistic details
{extra_strategy}

For each strategy, provide:
1. A brief name (3-5 words)
2. What textual patterns to look for
3. What to avoid (common false positives for this approach)

Output your strategies in this JSON format:
{{
    "strategies": [
        {{
            "name": "Strategy name",
            "focus": "What to focus on",
            "avoid": "What to avoid"
        }}
    ]
}}
"""

tot_path_extraction_prompt = """
Now, using ONLY the following extraction strategy, extract {interactions_type} interactions:

STRATEGY: {strategy_name}
FOCUS ON: {strategy_focus}
AVOID: {strategy_avoid}

Apply this strategy systematically to the text. {confidence_prompt}
"""

tot_evaluation_prompt = """
You have extracted {interactions_type} interactions using a specific strategy.

Review your extracted relations and evaluate:
1. How many relations did you find?
2. How confident are you in each relation (based on textual evidence)?
3. Are there any potential false positives?

For each triple, assign a quality score from 1-10 where:
- 10 = Explicit, clear interaction with strong textual evidence
- 7-9 = Clear interaction but less explicit evidence
- 4-6 = Possible interaction but ambiguous
- 1-3 = Weak evidence, likely false positive

Output in this JSON format:
{{
    "evaluation": [
        {{
            "head": "protein1",
            "relation": "INTERACTS_WITH",
            "tail": "protein2",
            "score": 9,
            "evidence": "Brief quote from text supporting this relation"
        }}
    ],
    "summary": "Brief assessment of this extraction path"
}}
"""

tot_merge_prompt = """
You have extracted {interactions_type} interactions using {n_paths} different reasoning strategies.

Below are the results from each strategy with quality scores:

{all_extractions}

Now, combine these results using the following approach:
- Include relations that appear in multiple strategies (higher confidence)
- Include relations with score ≥ 8 even if only found by one strategy
- Exclude relations with score < 5 unless they appear in ≥2 strategies
- Resolve conflicts (e.g., different relation types for same entity pair)

Output the final merged set of relations using the standard OUTPUT FORMAT.
{confidence_prompt}
"""
