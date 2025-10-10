import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import (  # isort:skip
    Entities,
    Message,
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
    tot_path_extraction_prompt,
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

string_db = {}
if args.string_db:
    # Load STRING database for knowledge decoration
    string_db_path = "/prj/LINDA_LLM/resources/string_interactions.tsv"  # Placeholder path; update as needed
    try:
        with open(string_db_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    protein1, protein2, score, interaction_type = (
                        parts[0],
                        parts[1],
                        float(parts[2]),
                        parts[3],
                    )
                    pair = tuple(sorted([protein1, protein2]))
                    string_db[pair] = {"score": score, "type": interaction_type}
        print(f"Loaded {len(string_db)} STRING interactions.")
    except FileNotFoundError:
        print(
            f"STRING database not found at {string_db_path}. Proceeding without STRING decoration."
        )

if args.dynex_k > 0:
    from datasets import concatenate_datasets

    from dataset import get_dataset
    from embed import client, embed_model, embeddings_path, load_index

    index = load_index()
    embeddings = np.load(embeddings_path)
    train_dataset, dev_dataset, _ = get_dataset(args.target, args.data)
    lookup_dataset = concatenate_datasets([train_dataset, dev_dataset])

print(f"New run: {triple_jsonl_path.parent}")
print(f"Len texts: {len(texts)}")

if args.all_ners_given:
    ner_paths = all_ner_paths
if args.true_ners_given:
    ner_paths = true_ner_paths


def get_string_info(proteins):
    """Fetch STRING info for a list of proteins (bonus: decorate with PPI knowledge)."""
    info = []
    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            pair = tuple(sorted([proteins[i], proteins[j]]))
            if pair in string_db:
                data = string_db[pair]
                info.append(
                    f"STRING PPI ({pair[0]} ↔ {pair[1]}): Score {data['score']:.2f} ({data['type']})"
                )
    return " | ".join(info) if info else "No STRING data available."


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


def extract_rels(messages, responses, text, prompts, examples_content=""):
    """Standard single-pass extraction"""
    for i, prompt in enumerate(prompts):
        content = f"\nUSER QUESTION: {prompt}"
        if examples_content:
            content += f"\n{examples_content}"
        messages.append(Message(role="user", content=content))
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
    messages,
    responses,
    text,
    prompts,
    n_samples=5,
    temperature=0.7,
    examples_content="",
):
    """Self-consistency ensemble extraction with voting"""
    print(f"  Running ensemble extraction with n={n_samples}, temp={temperature}")

    for i, prompt in enumerate(prompts):
        all_triples = []

        # Generate n predictions with temperature > 0
        for sample_idx in range(n_samples):
            # Create a copy of messages for each sample
            messages_copy = messages.copy()
            content = f"\nUSER QUESTION: {prompt}"
            if examples_content:
                content += f"\n{examples_content}"
            messages_copy.append(Message(role="user", content=content))

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
    """Enhanced RAG: Retrieve top-k diverse examples, format nicely, and decorate with STRING knowledge."""
    k = args.dynex_k  # Number of examples to retrieve; configurable via args
    embed = client.embed(
        model=embed_model,
        input=text,
    ).embeddings

    # Retrieve 2x candidates for diversity filtering
    labels, distances = index.knn_query(embed, k=k * 2)

    # Diversity filtering: Select k diverse examples using embedding distance
    selected_examples = []
    selected_embeds = []

    for idx, dist in zip(labels[0], distances[0]):
        example = lookup_dataset[int(idx)]
        example_embed = embeddings[int(idx)]

        # Skip if too similar to already selected (cosine similarity > 0.8, i.e., distance < 0.2)
        if any(np.dot(example_embed, sel_embed) > 0.8 for sel_embed in selected_embeds):
            continue

        selected_examples.append(example)
        selected_embeds.append(example_embed)

        if len(selected_examples) == k:
            break

    examples_text = ""
    for i, example in enumerate(selected_examples, start=1):
        example_doc = example["doc"]
        example_triples = example["triples"]

        example_line = (
            f"EXAMPLE {i}:\nText: {example_doc}\nRelations: {example_triples}"
        )
        if string_db:
            # Extract proteins from doc/triples for STRING lookup (simple regex; refine as needed)
            proteins = re.findall(
                r"\b[A-Z]{2,}\b", example_doc + " " + example_triples
            )  # E.g., match uppercase protein names
            string_info = get_string_info(proteins)
            example_line += f"\nKnowledge: {string_info}"
        example_line += "\n\n"
        examples_text += example_line

    messages.append(Message(role="user", content=f"SIMILAR EXAMPLES:\n{examples_text}"))


def extract_rels_tot(
    messages, responses, text, prompts, n_paths=3, strategy="vote", examples_content=""
):
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
            content = f"\n{prompt}\n\n{extraction_prompt}"
            if examples_content:
                content += f"\n{examples_content}"
            path_messages.append(Message(role="user", content=content))

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
    examples_content = ""  # Store examples content for inclusion in prompts
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
        if args.dynex_k > 0:
            get_dynex(messages, text)
            examples_content = messages[-1].content  # Get the examples content
            messages.pop()  # Remove it from messages, will be included in prompts

        # Choose extraction method based on flags
        if args.tot:
            extract_rels_tot(
                messages,
                responses,
                text,
                _prompts,
                n_paths=args.tot,
                strategy=args.tot_strategy,
                examples_content=examples_content,
            )
        elif args.ensemble:
            extract_rels_ensemble(
                messages,
                responses,
                text,
                _prompts,
                n_samples=args.ensemble,
                temperature=args.ensemble_temp,
                examples_content=examples_content,
            )
        else:
            extract_rels(
                messages, responses, text, _prompts, examples_content=examples_content
            )

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
