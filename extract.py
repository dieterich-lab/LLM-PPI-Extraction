import json
import os
import sys
from pathlib import Path

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import (  # isort:skip
    Entities,
    ExtractionStrategies,
    Message,
    PathEvaluation,
    Triples,
)
from baml_py import Collector

from baml.baml_client.type_builder import TypeBuilder
from clients import cr
from documents import all_ner_paths, texts, true_ner_paths
from paths import triple_jsonl_path, uniprot_path
from prompts import (
    confidence_prompt,
    interactions_type,
    prompts,
    rel_system_prompt,
    tot_evaluation_prompt,
    tot_merge_prompt,
    tot_path_extraction_prompt,
    tot_strategy_generation_prompt,
)

collector = Collector(name="my-collector")
tb = TypeBuilder()
if not args.noconfidence:
    tb.Triple.add_property(
        "confidence", tb.union([tb.literal_string("high"), tb.literal_string("low")])
    ).description("if this relation was extracted with high confidence or not")

if args.chattype == "lookup":
    with open(uniprot_path, "r") as lookupfile:
        lookup_table = {
            line.split("\t")[0]: (line.split("\t")[1], line.split("\t")[2])
            for line in lookupfile.readlines()[1:]
        }

if args.dynex:
    from datasets import concatenate_datasets

    from dataset import get_dataset
    from embed import client, embed_model, load_index

    index = load_index()
    train_dataset, dev_dataset, _ = get_dataset()
    lookup_dataset = concatenate_datasets([train_dataset, dev_dataset])

print(f"New run: {triple_jsonl_path.parent}")
print(f"Len texts: {len(texts)}")

if args.all_ners_given:
    ner_paths = all_ner_paths
if args.true_ners_given:
    ner_paths = true_ner_paths


def get_ners(messages, responses, doc, prompts):
    ner_prompt = prompts.pop(0)
    message = Message(role="user", content=ner_prompt)
    messages.append(message)
    try:
        ner_path = [
            x for x in ner_paths if doc[0].metadata["file_path"].stem == x.stem
        ][0]
        if ner_path:
            ners = open(ner_path, "r").readlines()
            ners = [x.strip() for x in ners]
        else:
            ners = []
        response = Entities(entities=ners)
    except:
        print(f"Exception at Entity extraction")
        response = Entities(entities=[])
    responses.append(response)
    messages.append(Message(role="assistant", content=f"{str(response)}"))
    return prompts


def extract_ners(messages, responses, text, prompts):
    ner_prompt = prompts.pop(0)
    message = Message(role="user", content=ner_prompt)
    messages.append(message)
    try:
        response = b.ExtractNEs(
            rel_system_prompt,
            text,
            message,
            baml_options={"client_registry": cr, "tb": tb, "collector": collector},
        )
    except:
        print(f"Exception at Entity extraction")
        response = Entities(entities=[])
    responses.append(response)
    messages.append(Message(role="assistant", content=f"{str(response)}"))
    return prompts


def extract_rels(messages, responses, text, prompts):
    """Standard single-pass extraction"""
    for i, prompt in enumerate(prompts):
        messages.append(Message(role="user", content=f"\nUSER QUESTION: {prompt}"))
        try:
            response = b.GeneralChatExtractRelationships(
                rel_system_prompt,
                text,
                messages,
                baml_options={"client_registry": cr, "tb": tb, "collector": collector},
            )
        except Exception as e:
            print(f"Exception at step {i}")
            response = Triples(triples=[])
        responses.append(response)
        messages.append(Message(role="assistant", content=str(response)))


def extract_rels_ensemble(
    messages, responses, text, prompts, n_samples=5, temperature=0.7
):
    """Self-consistency ensemble extraction with voting"""
    print(f"  Running ensemble extraction with n={n_samples}, temp={temperature}")

    for i, prompt in enumerate(prompts):
        all_triples = []

        # Generate n predictions with temperature > 0
        for sample_idx in range(n_samples):
            # Create a copy of messages for each sample
            messages_copy = messages.copy()
            messages_copy.append(
                Message(role="user", content=f"\nUSER QUESTION: {prompt}")
            )

            try:
                response = b.GeneralChatExtractRelationships(
                    rel_system_prompt,
                    text,
                    messages_copy,
                    baml_options={
                        "client_registry": cr,
                        "tb": tb,
                        "collector": collector,
                        "temperature": temperature,
                    },
                )
                all_triples.extend(response.triples)
                print(
                    f"    Sample {sample_idx + 1}/{n_samples}: {len(response.triples)} triples"
                )
            except Exception as e:
                print(f"    Exception at sample {sample_idx + 1}: {e}")
                continue

        # Vote: keep triples appearing in ≥50% of samples
        triple_counts = {}
        for triple in all_triples:
            # Create a unique key for each triple (case-insensitive to handle variations)
            key = (
                f"{triple.head.lower()}|{triple.relation.lower()}|{triple.tail.lower()}"
            )
            if key not in triple_counts:
                triple_counts[key] = {"count": 0, "example": triple}
            triple_counts[key]["count"] += 1

        # Select triples that appear in at least half of the samples
        threshold = n_samples // 2
        consensus_triples = [
            data["example"]
            for key, data in triple_counts.items()
            if data["count"] >= threshold
        ]

        print(
            f"    Consensus: {len(consensus_triples)} triples (threshold: {threshold}/{n_samples})"
        )

        # Create response with consensus triples
        consensus_response = Triples(triples=consensus_triples)
        responses.append(consensus_response)
        messages.append(Message(role="user", content=f"\nUSER QUESTION: {prompt}"))
        messages.append(Message(role="assistant", content=str(consensus_response)))


def lookup_infos(messages, responses):
    infos = dict()
    ent_set = set()
    for nes in responses[-1]:
        for ne in nes[1]:
            if ne in ent_set:
                continue
            ent_set.add(ne)
            if ne.lower() in lookup_table:
                infos[ne] = f"Function: {lookup_table[ne.lower()][0].strip()}"
    messages.append(Message(role="user", content=f"BACKGROUND KNOWLEDGE: {infos}\n"))


def get_dynex(messages, text):
    embed = client.embed(
        model=embed_model,
        input=text,
    ).embeddings
    labels, _ = index.knn_query(embed, k=1)
    example = lookup_dataset[labels[0]]
    example_text = "\n".join([example["doc"][0], example["triples"][0]])
    messages.append(Message(role="user", content=f"EXAMPLE: {example_text}\n"))


def extract_rels_tot(messages, responses, text, prompts, n_paths=3, strategy="vote"):
    """
    Tree-of-Thoughts extraction with multiple reasoning paths.

    Args:
        messages: Message history
        responses: Response list to append to
        text: Document text
        prompts: Extraction prompts
        n_paths: Number of reasoning paths to explore (default=3)
        strategy: How to combine results - 'vote', 'best', or 'merge'
    """
    print(f"  Running ToT extraction with n={n_paths} paths, strategy={strategy}")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"  ToT Step {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")

        # Step 1: Generate reasoning strategies using BAML
        print(f"    Generating {n_paths} reasoning strategies via LLM...")

        task_description = (
            f"Extract {interactions_type} interactions from biomedical text"
        )

        try:
            strategies_response = b.GenerateToTStrategies(
                task_description=task_description,
                n_paths=n_paths,
                text=text[:1000],  # Use first 1000 chars for strategy generation
                baml_options={"client_registry": cr, "tb": tb, "collector": collector},
            )
            strategies = [
                {"name": s.name, "focus": s.focus, "avoid": s.avoid}
                for s in strategies_response.strategies
            ]
            print(f"    Generated {len(strategies)} strategies successfully")
        except Exception as e:
            print(f"    Error generating strategies via LLM: {e}, using defaults")
            strategies = generate_default_strategies(n_paths)

        # Step 2: Extract relations using each strategy
        all_path_results = []
        all_path_evaluations = []

        for path_idx, strategy_dict in enumerate(strategies):
            print(f"    Path {path_idx + 1}/{n_paths}: {strategy_dict['name']}")

            # Extract using this strategy
            # Use messages.copy() to prevent each path from polluting other paths' message history
            path_messages = messages.copy()
            extraction_prompt = tot_path_extraction_prompt.format(
                interactions_type=interactions_type,
                strategy_name=strategy_dict["name"],
                strategy_focus=strategy_dict["focus"],
                strategy_avoid=strategy_dict["avoid"],
                confidence_prompt=confidence_prompt,
            )
            path_messages.append(
                Message(role="user", content=f"\n{prompt}\n\n{extraction_prompt}")
            )

            try:
                path_response = b.GeneralChatExtractRelationships(
                    rel_system_prompt,
                    text,
                    path_messages,
                    baml_options={
                        "client_registry": cr,
                        "tb": tb,
                        "collector": collector,
                    },
                )
                print(f"      Extracted: {len(path_response.triples)} triples")
                all_path_results.append(path_response.triples)

                # Step 3: Evaluate this path using BAML
                try:
                    path_eval_response = b.EvaluateToTPath(
                        text=text,
                        extracted_triples=path_response,
                        strategy_name=strategy_dict["name"],
                        task_description=task_description,
                        baml_options={
                            "client_registry": cr,
                            "tb": tb,
                            "collector": collector,
                        },
                    )

                    # Convert BAML evaluation to our internal format
                    path_eval = []
                    for eval_triple in path_eval_response.evaluated_triples:
                        # Find matching triple from extraction
                        matching_triple = None
                        for triple in path_response.triples:
                            if (
                                triple.head.lower() == eval_triple.head.lower()
                                and triple.tail.lower() == eval_triple.tail.lower()
                            ):
                                matching_triple = triple
                                break

                        if matching_triple:
                            path_eval.append(
                                {
                                    "triple": matching_triple,
                                    "score": eval_triple.score,
                                    "evidence": eval_triple.evidence,
                                    "path": path_idx,
                                }
                            )

                    print(
                        f"      Evaluated: avg score = {sum(e['score'] for e in path_eval) / len(path_eval):.1f}"
                        if path_eval
                        else "      Evaluated: no matches"
                    )
                    all_path_evaluations.append(path_eval)

                except Exception as e:
                    print(f"      Evaluation failed: {e}, using default scores")
                    # Fallback to simple scoring based on confidence attribute
                    path_eval = []
                    for triple in path_response.triples:
                        score = (
                            8
                            if not hasattr(triple, "confidence")
                            or triple.confidence == "high"
                            else 5
                        )
                        path_eval.append(
                            {"triple": triple, "score": score, "path": path_idx}
                        )
                    all_path_evaluations.append(path_eval)

            except Exception as e:
                print(f"      Exception in path {path_idx + 1}: {e}")
                all_path_results.append([])
                all_path_evaluations.append([])

        # Step 4: Combine results based on strategy
        if strategy == "vote":
            # Use default threshold (majority = ceil(n/2))
            final_triples = combine_by_voting(all_path_results)
        elif strategy == "best":
            final_triples = combine_by_best_path(all_path_results, all_path_evaluations)
        elif strategy == "merge":
            final_triples = combine_by_merging(all_path_results, all_path_evaluations)
        else:
            # Default to voting
            final_triples = combine_by_voting(all_path_results)

        print(f"    Final result: {len(final_triples)} triples")

        # Create final response
        final_response = Triples(triples=final_triples)
        responses.append(final_response)
        messages.append(Message(role="user", content=f"\nUSER QUESTION: {prompt}"))
        messages.append(Message(role="assistant", content=str(final_response)))


def generate_default_strategies(n_paths):
    """Generate default extraction strategies for ToT"""
    all_strategies = [
        {
            "name": "Explicit Interaction Verbs",
            "focus": "Look for explicit molecular interaction verbs like 'binds', 'phosphorylates', 'activates', 'inhibits', 'methylates', 'ubiquitinates', 'forms complex with'",
            "avoid": "General co-occurrence, pathway membership, or functional similarity without direct interaction evidence",
        },
        {
            "name": "Experimental Evidence",
            "focus": "Look for experimental methods that demonstrate direct interactions: co-immunoprecipitation, pull-down assays, Y2H, surface plasmon resonance, crosslinking, co-localization studies",
            "avoid": "Correlative evidence, expression patterns, or genetic interactions without biochemical validation",
        },
        {
            "name": "Mechanistic Details",
            "focus": "Look for mechanistic descriptions of how proteins interact: domain-domain interactions, substrate-enzyme relationships, complex stoichiometry, specific residues/sites involved",
            "avoid": "High-level functional relationships or regulatory effects that may be indirect",
        },
        {
            "name": "Signaling Cascade Context",
            "focus": "Look for interactions explicitly described within signaling pathways: upstream-downstream relationships, cascade components, scaffold proteins, adaptor proteins",
            "avoid": "Proteins merely mentioned together in pathway context without direct interaction evidence",
        },
        {
            "name": "Post-translational Modifications",
            "focus": "Look for PTM relationships: kinase-substrate, E3 ligase-target, methyltransferase-substrate, deubiquitinase-target, with specific modification sites when mentioned",
            "avoid": "Indirect regulatory effects or transcriptional regulation",
        },
    ]

    return all_strategies[:n_paths]


def combine_by_voting(all_path_results, threshold=None):
    """Combine ToT paths by majority voting

    Args:
        all_path_results: List of triple lists from each path
        threshold: Minimum number of paths that must agree (default: ceil(n/2) for majority)
    """
    import math

    triple_counts = {}

    for path_triples in all_path_results:
        for triple in path_triples:
            key = (
                f"{triple.head.lower()}|{triple.relation.lower()}|{triple.tail.lower()}"
            )
            if key not in triple_counts:
                triple_counts[key] = {"count": 0, "example": triple}
            triple_counts[key]["count"] += 1

    # Default threshold: majority (ceil(n_paths / 2))
    # For n=3: ceil(3/2) = 2 (not 1!)
    # For n=4: ceil(4/2) = 2
    # For n=5: ceil(5/2) = 3
    if threshold is None:
        n_paths = len(all_path_results)
        threshold = math.ceil(n_paths / 2)

    # Keep triples appearing in at least 'threshold' paths
    consensus_triples = [
        data["example"]
        for key, data in triple_counts.items()
        if data["count"] >= threshold
    ]

    return consensus_triples


def combine_by_best_path(all_path_results, all_path_evaluations):
    """Select results from the highest-scored path"""
    if not all_path_evaluations or not any(all_path_evaluations):
        return all_path_results[0] if all_path_results else []

    # Calculate average score for each path
    path_scores = []
    for path_eval in all_path_evaluations:
        if path_eval:
            avg_score = sum(item["score"] for item in path_eval) / len(path_eval)
            path_scores.append(avg_score)
        else:
            path_scores.append(0)

    # Return triples from best path
    best_path_idx = path_scores.index(max(path_scores)) if path_scores else 0
    return all_path_results[best_path_idx]


def combine_by_merging(all_path_results, all_path_evaluations):
    """Merge high-confidence triples from all paths"""
    merged_triples = {}

    for path_idx, path_eval in enumerate(all_path_evaluations):
        for item in path_eval:
            triple = item["triple"]
            score = item["score"]
            key = (
                f"{triple.head.lower()}|{triple.relation.lower()}|{triple.tail.lower()}"
            )

            # Include if: score >= 8, OR appears in multiple paths, OR score >= 6 and in 2+ paths
            if key not in merged_triples:
                merged_triples[key] = {"triple": triple, "max_score": score, "count": 1}
            else:
                merged_triples[key]["max_score"] = max(
                    merged_triples[key]["max_score"], score
                )
                merged_triples[key]["count"] += 1

    # Filter based on confidence criteria
    final_triples = [
        data["triple"]
        for key, data in merged_triples.items()
        if data["max_score"] >= 8
        or data["count"] >= 2
        or (data["max_score"] >= 6 and data["count"] >= 2)
    ]

    return final_triples


def main():
    if args.force_new:
        with open(triple_jsonl_path, "w") as f:  # delete the content
            pass

    # Apply document slicing based on startfromdoc and untildoc
    docs_to_process = texts
    if args.untildoc > 0:
        docs_to_process = texts[args.startfromdoc : args.untildoc]
    elif args.startfromdoc > 0:
        docs_to_process = texts[args.startfromdoc :]

    print(
        f"Processing documents {args.startfromdoc} to {args.untildoc if args.untildoc > 0 else len(texts)} (total: {len(docs_to_process)})"
    )

    for i, doc in enumerate(docs_to_process, start=args.startfromdoc):
        file_path = doc[0].metadata["file_path"]
        if not args.force_new and not args.dev:
            try:
                with open(triple_jsonl_path, "r") as f:
                    if any(
                        Path(json.loads(line)["filename"]).stem == file_path.stem
                        for line in f
                    ):
                        print(f"Skipping {file_path.stem}")
                        continue
            except FileNotFoundError:
                pass
        _prompts = prompts.copy()
        print(f"Doc {i}")
        text = doc[0].page_content
        messages = list()
        responses = list()
        if args.all_ners_given or args.true_ners_given:
            _prompts = get_ners(messages, responses, doc, _prompts)
        elif args.extractionmode == "nerrel":
            _prompts = extract_ners(messages, responses, text, doc, _prompts)
        if args.chattype == "lookup":
            lookup_infos(messages, responses)
        if args.dynex:
            get_dynex(messages, text)

        # Choose extraction method based on flags
        if args.tot:
            extract_rels_tot(
                messages,
                responses,
                text,
                _prompts,
                n_paths=args.tot,
                strategy=args.tot_strategy,
            )
        elif args.ensemble:
            extract_rels_ensemble(
                messages,
                responses,
                text,
                _prompts,
                n_samples=args.ensemble,
                temperature=args.ensemble_temp,
            )
        else:
            extract_rels(messages, responses, text, _prompts)

        if not args.dev:
            result = {
                "responses": [r.model_dump()["triples"] for r in responses],
                "text": doc[0].page_content,
                "filename": str(doc[0].metadata["file_path"]),
            }
            with open(triple_jsonl_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
