import json
import os
import pickle
from parser import args
from pathlib import Path

from get_documents import documents, paper_dict
from json_repair import repair_json
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm import llm, model
from prompt_utils import create_unstructured_prompt
from structured_classes import (
    PPI_Triples,
    PPI_Triples_Simple,
    TF_Triples,
    TF_Triples_Simple,
    Triples,
    Triples_Simple,
)
from style_templates import style_dict
from templates import (
    PPI_BASESTRINGPARTS,
    PPI_BASESTRINGPARTS_SIMPLE,
    PPI_EXAMPLES,
    PPI_EXAMPLES_SIMPLE,
    PPI_INTERACTIONS,
    PPI_NODE_LABELS,
    TEMPLATE,
    TEMPLATE_SIMPLE,
    TF_BASESTRINGPARTS,
    TF_BASESTRINGPARTS_SIMPLE,
    TF_EXAMPLES,
    TF_EXAMPLES_SIMPLE,
    TF_INTERACTIONS,
    TF_NODE_LABELS,
)
from utils import Timeout

basestring_dict = {"ppi": PPI_BASESTRINGPARTS, "tf": TF_BASESTRINGPARTS}
simple_basestring_dict = {
    "ppi": PPI_BASESTRINGPARTS_SIMPLE,
    "tf": TF_BASESTRINGPARTS_SIMPLE,
}
example_dict = {
    "ppi": PPI_EXAMPLES,
    "tf": TF_EXAMPLES,
    "both": PPI_EXAMPLES + TF_EXAMPLES,
}
simple_example_dict = {
    "ppi": PPI_EXAMPLES_SIMPLE,
    "tf": TF_EXAMPLES_SIMPLE,
    "both": PPI_EXAMPLES_SIMPLE + TF_EXAMPLES_SIMPLE,
}
interactions_dict = {
    "ppi": PPI_INTERACTIONS,
    "tf": TF_INTERACTIONS,
    "both": PPI_INTERACTIONS + TF_INTERACTIONS,
}
nodelabels_dict = {
    "ppi": PPI_NODE_LABELS,
    "tf": TF_NODE_LABELS,
    "both": PPI_NODE_LABELS + TF_NODE_LABELS,
}

schema_dict = {"ppi": PPI_Triples, "tf": TF_Triples, "both": Triples}
simple_schema_dict = {
    "ppi": PPI_Triples_Simple,
    "tf": TF_Triples_Simple,
    "both": Triples_Simple,
}
if not args.style:
    basestring_parts = (
        basestring_dict[args.target]
        if not args.simple
        else simple_basestring_dict[args.target]
    )
else:
    mode = "simple" if args.simple else "complex"
    basestring_parts = style_dict[args.style][mode][args.target][0]

schema = (
    schema_dict[args.target] if not args.simple else simple_schema_dict[args.target]
)
# if args.target != "both":
if not args.style:
    prompt = create_unstructured_prompt(
        base_string_parts=basestring_parts,
        template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
        examples=(
            example_dict[args.target]
            if not args.simple
            else simple_example_dict[args.target]
        ),
        schema=schema,
        node_labels=False if args.simple else nodelabels_dict[args.target],
        rel_types=interactions_dict[args.target],
    )
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[] if args.simple else nodelabels_dict[args.target],
        allowed_relationships=[] if args.simple else interactions_dict[args.target],
        strict_mode=not args.simple,
        prompt=prompt,
    )
else:
    system_message = SystemMessage(content=basestring_parts)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    parser = JsonOutputParser(pydantic_object=schema)

    human_prompt = PromptTemplate(
        template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": False if args.simple else nodelabels_dict[args.target],
            "rel_types": interactions_dict[args.target],
            "examples": (
                example_dict[args.target]
                if not args.simple
                else simple_example_dict[args.target]
            ),
            "previous_examples": None,
        },
    )
# else:
#     ppi_prompt = create_unstructured_prompt(
#         base_string_parts=basestring_parts,
#         template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
#         examples=(PPI_EXAMPLES if not args.simple else PPI_EXAMPLES_SIMPLE),
#         schema=schema,
#         node_labels=False if args.simple else PPI_NODE_LABELS,
#         rel_types=False if args.simple else PPI_INTERACTIONS,
#     )
#     ppi_llm_transformer = LLMGraphTransformer(
#         llm=llm,
#         allowed_nodes=[] if args.simple else PPI_NODE_LABELS,
#         allowed_relationships=[] if args.simple else PPI_INTERACTIONS,
#         strict_mode=not args.simple,
#         prompt=ppi_prompt,
#     )

graph_doc_filename = "graph_documents.pkl"

graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{args.target}/{args.parser}/{model}/graph_documents.pkl"

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )
if args.style:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / f"style{args.style}" / graph_doc_filename
    )
if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename
else:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "complex" / graph_doc_filename

os.makedirs(Path(graphdoc_pkl_path).parent, exist_ok=True)

if args.target == "both":
    ppi_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "ppi_graph_documents.pkl"
    tf_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "tf_graph_documents.pkl"
    os.makedirs(Path(ppi_graphdoc_pkl_path).parent, exist_ok=True)
    os.makedirs(Path(tf_graphdoc_pkl_path).parent, exist_ok=True)

graph_documents = list()

if not args.dev:
    if args.target != "both":
        f = open(graphdoc_pkl_path, "wb")
    else:
        ppi_f = open(ppi_graphdoc_pkl_path, "wb")
        tf_f = open(tf_graphdoc_pkl_path, "wb")

if args.style:
    structured_llm = llm.with_structured_output(schema, include_raw=True)
    graph_runnable = prompt | structured_llm
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        response = graph_runnable.invoke({"messages": state["messages"]})
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

# for i, (doc, id) in enumerate(documents[320:]):
for i, (doc, id) in enumerate(documents):
    print(i, id)
    c = 0
    try:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        while c < 5:
            try:
                with Timeout(180):
                    if not args.style:
                        graph_doc = llm_transformer.convert_to_graph_documents([doc])[0]
                    else:
                        config = {"configurable": {"thread_id": id}}
                        msg = app.invoke(
                            {
                                "messages": [
                                    HumanMessage(
                                        human_prompt.format(input=doc.page_content)
                                    )
                                ]
                            },
                            config,
                        )
                        if args.target != "both":
                            for human_msg in style_dict[args.style][mode][args.target][
                                1:
                            ]:
                                msg = app.invoke(
                                    {"messages": [HumanMessage(human_msg)]},
                                    config,
                                )
                            triples = json.loads(msg["messages"][-1].content)["triples"]
                        else:
                            msg = app.invoke(
                                {
                                    "messages": [
                                        HumanMessage(
                                            style_dict[args.style][mode][args.target][1]
                                        )
                                    ]
                                },
                                config,
                            )
                            if args.style == 2:
                                ppi_triples = json.loads(msg["messages"][-1].content)[
                                    "triples"
                                ]
                            elif args.style == 2:
                                tf_triples = json.loads(msg["messages"][-1].content)[
                                    "triples"
                                ]
                            msg = app.invoke(
                                {
                                    "messages": [
                                        HumanMessage(
                                            style_dict[args.style][mode][args.target][2]
                                        )
                                    ]
                                },
                                config,
                            )
                            if args.style == 2:
                                tf_triples = json.loads(msg["messages"][-1].content)[
                                    "triples"
                                ]
                            elif args.style == 2:
                                ppi_triples = json.loads(msg["messages"][-1].content)[
                                    "triples"
                                ]
                    break
            except Timeout.Timeout:
                print("Timeout")
                c += 1
        if args.target != "both":
            nodes_set = set()
            rels = list()
            for triple in triples:
                if isinstance(triples, str):
                    triple = json.loads(repair_json(triple))
                n1 = triple["head"]
                n2 = triple["tail"]
                nodes_set.add(n1)
                nodes_set.add(n2)
                rels.append(
                    Relationship(
                        source=Node(id=n1), target=Node(id=n2), type=triple["relation"]
                    )
                )
            nodes = [Node(id=el) for el in list(nodes_set)]
            graph_doc = GraphDocument(nodes=nodes, relationships=rels, source=doc)
            graph_doc.source.metadata["source"] = paper_dict[id]
            graph_doc.source.metadata["id"] = str(id)
            if not args.dev:
                pickle.dump(graph_doc, f)
        else:
            task_triples = [ppi_triples, tf_triples]
            if not args.dev:
                task_files = [ppi_f, tf_f]
            else:
                task_files = [None, None]
            for triples, f in zip(task_triples, task_files):
                nodes_set = set()
                rels = list()
                for triple in triples:
                    if isinstance(triples, str):
                        triple = json.loads(repair_json(triple))
                    n1 = triple["head"]
                    n2 = triple["tail"]
                    nodes_set.add(n1)
                    nodes_set.add(n2)
                    rels.append(
                        Relationship(
                            source=Node(id=n1),
                            target=Node(id=n2),
                            type=triple["relation"],
                        )
                    )
                nodes = [Node(id=el) for el in list(nodes_set)]
                graph_doc = GraphDocument(nodes=nodes, relationships=rels, source=doc)
                graph_doc.source.metadata["source"] = paper_dict[id]
                graph_doc.source.metadata["id"] = str(id)
                if not args.dev:
                    pickle.dump(graph_doc, f)
    except Exception as e:
        print(e)

if not args.dev:
    if args.target != "both":
        f.close()
    else:
        ppi_f.close()
        tf_f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
