"""
Refactored prompts.py - A more maintainable and pythonic approach to prompt management.
"""

from dataclasses import dataclass
from parser import args
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionConfig:
    """Configuration for extraction settings."""

    target: str  # "ppi", "tf", or "ppitf"
    extraction_mode: str  # "direct" or "nerrel"
    chat_type: str  # "oneshot" or "stepwise"
    model: str
    force_cot: bool
    noconfidence: bool
    recall: bool
    examples: str  # "negpos", "pos", "neg", or ""
    all_nes_given: bool
    true_nes_given: bool
    spacy_nes_given: bool
    lookup: bool
    dynex_k: int

    @classmethod
    def from_args(cls) -> "ExtractionConfig":
        """Create config from global args."""
        return cls(
            target=args.target,
            extraction_mode=args.extractionmode,
            chat_type=args.chattype,
            model=args.model,
            force_cot=args.force_cot,
            noconfidence=args.noconfidence,
            recall=args.recall,
            examples=args.examples,
            all_nes_given=args.all_nes_given,
            true_nes_given=args.true_nes_given,
            spacy_nes_given=args.spacy_nes_given,
            lookup=args.lookup,
            dynex_k=args.dynex_k,
        )


@dataclass
class TargetConfig:
    """Configuration for different target types."""

    targets: str
    anti_targets: str
    target: str
    interactions_type: str
    anti_interactions_type: str

    @classmethod
    def for_target(cls, target: str) -> "TargetConfig":
        """Factory method to create target config based on target type."""
        configs = {
            "ppi": cls(
                targets="proteins",
                anti_targets="transcription factors and genes",
                target="protein",
                interactions_type="protein-protein",
                anti_interactions_type="transcription factor/gene",
            ),
            "tf": cls(
                targets="transcription factors and genes",
                anti_targets="proteins",
                target="transcription factor/gene",
                interactions_type="transcription factor-to-gene",
                anti_interactions_type="protein-protein",
            ),
            "ppitf": cls(
                targets="proteins and transcription factors and genes",
                anti_targets="",
                target="protein and transcription factor/gene",
                interactions_type="protein-protein and transcription factor-to-gene",
                anti_interactions_type="",
            ),
        }
        return configs[target]


class ExamplesData:
    """Container for all example data."""

    PPI_POSITIVE = """
Below you find some positive examples of protein-protein interactions that give you an idea of what we are looking for:
* "This cytokine induces the p53 into a mutant-like conformation that forms a complex with Sp1": p53 -> Sp1
* "These findings suggest that the STAT3-NRF2 complex accelerates BLBC growth and progression by augmenting IL-23A expression.": STAT3 -> NRF2
* "HIF1A forms a transcriptional complex with ARNT under hypoxia.": HIF1A -> ARNT
* "PRMT1 methylates cGAS and suppresses cGAS/STING signaling in cancer cells": PRMT1 -> cGAS
* "TRAF6 ubiquitinates TGFβ type I receptor to promote its cleavage and nuclear translocation in cancer.": TRAF6 -> TGFβ
* "AKT1 phosphorylates AKT1S1 at Thr-246.": AKT1 -> AKT1S1
* "PIAS1 sumoylates PNKP in cells.": PIAS1 -> PNKP
* "CBP, but not p/CAF, acetylates GATA-1 at two highly conserved lysine-rich motifs present at the C-terminal tails of both zinc fingers.": CBP -> GATA-1
"""

    PPI_NEGATIVE = """
Below you find some examples of false positive protein-protein relations and the reason why you should not extract those:
* "KRAS and BRAF cooperate in the MAPK signaling cascade to promote cell proliferation.": Although the two proteins are in the same signalling system, the text does not provide evidence of a direct interaction.
* "p53 and Protein MYC are both found in the same signaling complex.": Incorrect assumptions based on co-occurrence or proximity.
* "TNF and IL6 accumulate at DNA damage sites.": Co-localization but no evidence of direct relation/interaction between the two.
* "Gene TNF regulates the expression of Gene IL6": Misinterpretation of genetic or signalling pathways as protein interactions.
* "Prmt5 shares 80% sequence identity with Protein Prmt7, which is known to bind BRAF.": Incorrect assumptions based on structural similarity.
* "PTEN was pulled down in a co-IP assay with CDKN2A.": Incorrect interpretations of experimental methods.
"""

    TF_POSITIVE = """
Below you find some positive examples of relations between transcription factor-to-gene relations that give you an idea of what we are looking for:
* "MYC target genes that are involved in cell cycle such as Cyclin D1": MYC -> Cyclin D1
* "STAT3 can induce the expression of anti-apoptotic genes like Bcl-2, which help in cell survival.": STAT3 -> Bcl-2
* "Tbx1 activates transcription of the fibroblast growth factor genes Fgf8 and Fgf10 to maintain proliferative expansion and inhibit differentiation of cardiopharyngeal precursor cells": Tbx1 -> Fgf8 and Tbx1 -> Fgf10
* "Overexpression of MECP2 leads to the suppression of IFN-γ transcription, which is linked to impaired TH1 responses in both children and mice with MECP2 duplication syndrome.": MECP2 -> IFN-γ
* "In humans, FOXO regulates the expression of core small RNA pathway genes, including AGO2.": FOXO -> AGO2.
"""

    TF_NEGATIVE = """
Below you find some false positive examples of transcription factor-to-gene relations and the reason why you should not extract those:
* "This cytokine induces the p53 into a mutant-like conformation that forms a complex with Sp1": This is complex formation and does not involve transcription factor to gene relations.
* "HIF1A forms a transcriptional complex with ARNT under hypoxia.": This relation represents two transcription factor proteins interacting with each other, and the text does not reflect that they target the regulation of any specific gene.
* "PRMT1 methylates cGAS and suppresses cGAS/STING signaling in cancer cells": This is a methylation interaction and not a transcription factor to gene relation.
* "Gene MYC and gene STAT3 share a common promoter region.": This is not explicitly a relation between a transcription factor and its target gene.
* "AKT1 phosphorylates AKT1S1 at Thr-246.": This is a phosphorylation interaction and not a transcription factor to gene relation.
"""

    @classmethod
    def get_examples(cls, target: str, example_type: str) -> str:
        """Get examples based on target and type."""
        examples_map = {
            ("ppi", "pos"): cls.PPI_POSITIVE,
            ("ppi", "neg"): cls.PPI_NEGATIVE,
            ("tf", "pos"): cls.TF_POSITIVE,
            ("tf", "neg"): cls.TF_NEGATIVE,
            ("ppitf", "pos"): cls.PPI_POSITIVE + "\n" + cls.TF_POSITIVE,
            ("ppitf", "neg"): cls.PPI_NEGATIVE + "\n" + cls.TF_NEGATIVE,
        }
        return examples_map.get((target, example_type), "")


class PromptBuilder:
    """Builds prompts based on configuration."""

    COT_MODELS = ["llama33", "llama31", "llama33regu", "llama31regu"]

    SYSTEM_PROMPT = (
        "You are a top-tier molecular biologist specialized in the field of cardiology and molecular biology. "
        "Following, you'll find a scientific TEXT, a desired OUTPUT FORMAT and a USER QUESTION. "
        "First, read the TEXT and study the OUTPUT FORMAT, then answer the USER QUESTION."
    )

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

    def __init__(self, config: ExtractionConfig, target_config: TargetConfig):
        self.config = config
        self.target_config = target_config

    def build_cot_prompt(self) -> str:
        """Build chain-of-thought prompt."""
        if self.config.model in self.COT_MODELS and self.config.force_cot:
            return "Let's think step by step. "
        return ""

    def build_confidence_prompt(self) -> str:
        """Build confidence prompt."""
        if not self.config.noconfidence:
            return "Whenever you are unsure about a relation set the 'confidence' attribute to 'low', otherwise to 'high'."
        return ""

    def build_ner_list_prompt(self) -> str:
        """Build NER list prompt based on configuration."""
        if self.config.all_nes_given:
            return (
                f"Look at the list above. These are the ground truth {self.target_config.targets} "
                f"that are found in the abstract. But be wary, not all of them will necessarily be a "
                f"participant in {self.target_config.interactions_type} relations. Use this list for the following task:"
            )
        elif self.config.true_nes_given:
            return (
                f"Look at the list above. These are the ground truth {self.target_config.targets} "
                f"that are found in the abstract and also take part in {self.target_config.interactions_type} "
                "relations. Use this list for the following task:"
            )
        elif self.config.spacy_nes_given:
            return (
                f"Look at the list above. These are {self.target_config.targets} that have been "
                "extracted by a ScispaCy biomedical NER model. Use this list for the following task:"
            )
        elif self.config.extraction_mode == "nerrel":
            if not (
                self.config.all_nes_given
                or self.config.true_nes_given
                or self.config.spacy_nes_given
            ):
                return f"Now look at your extracted {self.target_config.targets} above and use it for the following task:"
        return ""

    def build_ner_prompt(self) -> str:
        """Build NER prompt modifier."""
        if (
            self.config.true_nes_given
            or self.config.all_nes_given
            or self.config.spacy_nes_given
        ):
            return "for the entities in the NE LIST above that are "
        return ""

    def build_lookup_prompt(self) -> str:
        """Build lookup prompt."""
        if self.config.lookup:
            return "We also provided above some insightful BACKGROUND KNOWLEDGE for each extracted protein. Use it as additional support. "
        return ""

    def build_dynex_prompt(self) -> str:
        """Build dynamic example prompt."""
        if self.config.dynex_k > 0:
            return "Following, you find an EXAMPLE of a similar texts and ground truth relations. Use it as support for your decision. "
        return ""

    def build_main_prompt(self) -> str:
        """Build the main extraction prompt."""
        ner_list = self.build_ner_list_prompt()
        ner_modifier = self.build_ner_prompt()
        lookup = self.build_lookup_prompt()
        dynex = self.build_dynex_prompt()

        if not self.config.recall:
            return (
                f"{ner_list} Extract all the {self.target_config.interactions_type} interactions "
                f"{ner_modifier}involved in signalling pathways from the text. Please only extract "
                f"{self.target_config.target} pairs which directly interact with each other "
                "(i.e. through binding, phosphorylation, sumoylation, etc). Do not misinterpret "
                "functional relationships, co-occurrence, structural similarity, or indirect "
                f"regulatory effects for direct interactions. {lookup}{dynex}"
            )
        else:
            return (
                f"{ner_list} Extract ALL the relations between molecular entities from the text. "
                f"Be as greedy as possible, we will filter the relations for correctness later in a second step {lookup}"
            )

    def build_examples(self) -> str:
        """Build examples string based on configuration."""
        if self.config.target == "ppi":
            pos = (
                ExamplesData.get_examples("ppi", "pos")
                if self.config.examples in ["negpos", "pos"]
                else ""
            )
            neg = (
                ExamplesData.get_examples("ppi", "neg")
                if self.config.examples in ["negpos", "neg"]
                else ""
            )
            return pos + neg
        elif self.config.target == "tf":
            pos = (
                ExamplesData.get_examples("tf", "pos")
                if self.config.examples in ["negpos", "pos"]
                else ""
            )
            neg = (
                ExamplesData.get_examples("tf", "neg")
                if self.config.examples in ["negpos", "neg"]
                else ""
            )
            return pos + neg
        elif self.config.target == "ppitf":
            pos = (
                ExamplesData.get_examples("ppitf", "pos")
                if self.config.examples in ["negpos", "pos"]
                else ""
            )
            neg = (
                ExamplesData.get_examples("ppitf", "neg")
                if self.config.examples in ["negpos", "neg"]
                else ""
            )
            return pos + neg
        return ""

    def build_chat_prompts(self) -> List[str]:
        """Build the final chat prompts based on configuration."""
        main_prompt = self.build_main_prompt()
        examples = self.build_examples()
        confidence = self.build_confidence_prompt()
        cot = self.build_cot_prompt()

        # Determine mode
        mode = self.config.extraction_mode
        if self.config.all_nes_given or self.config.true_nes_given:
            mode = "nerrel"

        base_prompt = f"{main_prompt} {examples} Please stick to the desired OUTPUT FORMAT. {confidence}{cot}"

        # Build prompts based on mode and chat type
        if mode == "direct":
            if self.config.chat_type == "oneshot":
                return [base_prompt]
            elif self.config.chat_type == "stepwise":
                return [
                    base_prompt,
                    f"Now review your extracted {self.target_config.interactions_type} interactions to determine if "
                    "they are specific to signaling pathways. Retain only signalling pathway interactions "
                    "and remove the rest. "
                    "Use again the desired json OUTPUT FORMAT to format your answer. "
                    f"{confidence}{cot}",
                    f"Review one more time the {self.target_config.interactions_type} interactions to  "
                    f"determine whether there are in the list regulations that are of a {self.target_config.anti_interactions_type} "
                    f"regulatory nature. Retain those interactions that are only specific to {self.target_config.interactions_type} interactions in cell  "
                    f"signalling and remove those relations that represent relations between {self.target_config.anti_targets}. "
                    "Use again the desired json OUTPUT FORMAT to format your answer. "
                    f"{confidence}{cot}",
                ]
        elif mode == "nerrel":
            ner_prompt = f"Extract all the {self.target_config.targets} that appear in the text. Please stick to the desired OUTPUT FORMAT. "
            rel_prompt = base_prompt

            if self.config.chat_type == "oneshot":
                return [ner_prompt, rel_prompt]
            elif self.config.chat_type in ["stepwise", "lookup"]:
                return [
                    f"Extract all the {self.target_config.targets} that appear in the text.",
                    rel_prompt,
                    f"Now review your extracted {self.target_config.interactions_type} interactions to determine if "
                    "they are specific to signaling pathways. Retain only signalling pathway interactions "
                    "and remove the rest. "
                    "Use again the desired json OUTPUT FORMAT to format your answer. "
                    f"{confidence}{cot}",
                    f"Review one more time the {self.target_config.interactions_type} interactions to  "
                    f"determine whether there are in the list regulations that are of a {self.target_config.anti_interactions_type} "
                    f"regulatory nature. Retain those interactions that are only specific to {self.target_config.interactions_type} interactions in cell  "
                    f"signalling and remove those relations that represent relations between {self.target_config.anti_targets}. "
                    "Use again the desired json OUTPUT FORMAT to format your answer. "
                    f"{confidence}{cot}",
                ]

        # Handle special case for ppitf stepwise
        if self.config.target == "ppitf" and self.config.chat_type == "stepwise":
            prompts = self.build_chat_prompts()
            return prompts[:-1]  # Remove last prompt for ppitf stepwise

        return []


def create_prompt_builder() -> PromptBuilder:
    """Factory function to create a prompt builder from global args."""
    config = ExtractionConfig.from_args()
    target_config = TargetConfig.for_target(config.target)
    return PromptBuilder(config, target_config)


# ============================================================================
# Tree-of-Thoughts (ToT) Prompts
# ============================================================================

TOT_STRATEGY_GENERATION_PROMPT = """
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

TOT_PATH_EXTRACTION_PROMPT = """
Now, using ONLY the following extraction strategy, extract {interactions_type} interactions:

STRATEGY: {strategy_name}
FOCUS ON: {strategy_focus}
AVOID: {strategy_avoid}

Apply this strategy systematically to the text. {confidence_prompt}
"""

TOT_EVALUATION_PROMPT = """
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

TOT_MERGE_PROMPT = """
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


# ============================================================================
# Main Interface - Backward Compatibility
# ============================================================================

# Create the prompt builder instance
_builder = create_prompt_builder()

# Export the prompts and constants for backward compatibility
rel_system_prompt = PromptBuilder.SYSTEM_PROMPT
prompts = _builder.build_chat_prompts()
OUTPUT_FORMAT = PromptBuilder.OUTPUT_FORMAT

# ToT prompts (keeping original names for compatibility)
tot_strategy_generation_prompt = TOT_STRATEGY_GENERATION_PROMPT
tot_path_extraction_prompt = TOT_PATH_EXTRACTION_PROMPT
tot_evaluation_prompt = TOT_EVALUATION_PROMPT
tot_merge_prompt = TOT_MERGE_PROMPT
