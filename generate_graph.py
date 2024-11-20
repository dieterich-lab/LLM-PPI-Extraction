import pickle
from parser import args

from const import LOOKUP
from get_documents import documents, paper_dict, whole_documents
from graph_utils import parse_msg2triples
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm import llm
from paths import graphdoc_pkl_path, ppi_graphdoc_pkl_path, tf_graphdoc_pkl_path
from structured_classes import (
    LR_Triples_Simple,
    PPI_Triples,
    PPI_Triples_Simple,
    TF_Triples,
    TF_Triples_Simple,
    Triples,
    Triples_Simple,
)
from style_templates import style_dict
from templates import (
    LR_EXAMPLES_SIMPLE,
    LR_INTERACTIONS,
    LR_NODE_LABELS,
    PPI_EXAMPLES,
    PPI_EXAMPLES_SIMPLE,
    PPI_INTERACTIONS,
    PPI_NODE_LABELS,
    TEMPLATE,
    TEMPLATE_SIMPLE,
    TF_EXAMPLES,
    TF_EXAMPLES_SIMPLE,
    TF_INTERACTIONS,
    TF_NODE_LABELS,
)
from utils import Timeout

example_dict = {
    "ppi": PPI_EXAMPLES,
    "tf": TF_EXAMPLES,
    "both": PPI_EXAMPLES + TF_EXAMPLES,
}
simple_example_dict = {
    "ppi": PPI_EXAMPLES_SIMPLE,
    "tf": TF_EXAMPLES_SIMPLE,
    "lr": LR_EXAMPLES_SIMPLE,
    "both": PPI_EXAMPLES_SIMPLE + TF_EXAMPLES_SIMPLE,
}
interactions_dict = {
    "ppi": PPI_INTERACTIONS,
    "tf": TF_INTERACTIONS,
    "lr": LR_INTERACTIONS,
    "both": PPI_INTERACTIONS + TF_INTERACTIONS,
}
nodelabels_dict = {
    "ppi": PPI_NODE_LABELS,
    "tf": TF_NODE_LABELS,
    "tf": LR_NODE_LABELS,
    "both": PPI_NODE_LABELS + TF_NODE_LABELS,
}

schema_dict = {
    "ppi": PPI_Triples,
    "tf": TF_Triples,
    "both": Triples,
}

simple_schema_dict = {
    "ppi": PPI_Triples_Simple,
    "tf": TF_Triples_Simple,
    "lr": LR_Triples_Simple,
    "both": Triples_Simple,
}

mode = "simple" if args.simple else "complex"
basestring_parts = style_dict[args.style][mode][LOOKUP][0]

schema = schema_dict[LOOKUP] if not args.simple else simple_schema_dict[LOOKUP]

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
        "node_labels": False if args.simple else nodelabels_dict[LOOKUP],
        "rel_types": interactions_dict[LOOKUP],
        "examples": (
            example_dict[LOOKUP] if not args.simple else simple_example_dict[LOOKUP]
        ),
    },
)

f = None
ppi_f = None
tf_f = None
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


target_docs = documents if not args.doclevel else whole_documents
print(len(target_docs))

for i, (doc, id) in enumerate(target_docs[args.startfromdoc :], args.startfromdoc):
    print(i, id)
    c = 0
    # try:
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    while c < 5:
        try:
            with Timeout(1800):
                config = {"configurable": {"thread_id": id}}
                msg = app.invoke(
                    {
                        "messages": [
                            HumanMessage(human_prompt.format(input=doc.page_content))
                        ]
                    },
                    config,
                )
                if args.target != "both":
                    for human_msg in style_dict[args.style][mode][LOOKUP][1:]:
                        msg = app.invoke(
                            {"messages": [HumanMessage(human_msg)]},
                            config,
                        )
                    final_message = msg["messages"][-1]
                else:
                    msg = app.invoke(
                        {
                            "messages": [
                                HumanMessage(style_dict[args.style][mode][LOOKUP][1])
                            ]
                        },
                        config,
                    )
                    if args.style == 2:
                        ppi_final_message = msg["messages"][-1]
                    elif args.style == 3:
                        tf_final_message = msg["messages"][-1]
                    msg = app.invoke(
                        {
                            "messages": [
                                HumanMessage(style_dict[args.style][mode][LOOKUP][2])
                            ]
                        },
                        config,
                    )
                    if args.style == 2:
                        tf_final_message = msg["messages"][-1]
                    elif args.style == 3:
                        ppi_final_message = msg["messages"][-1]
                break
        except Timeout.Timeout:
            print("Timeout")
            c += 1
    nodes_set = set()
    rels = list()
    if args.target == "both" and (not ppi_final_message and not tf_final_message):
        continue
    if args.target != "both" and not final_message:
        continue

    final_messages = (
        [final_message]
        if args.target != "both"
        else [ppi_final_message, tf_final_message]
    )
    files = [f] if args.target != "both" else [ppi_f, tf_f]

    for final_message, file in zip(final_messages, files):
        triples = parse_msg2triples(final_message)
        for triple in triples:
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
        print(len(graph_doc.relationships))
        graph_doc.source.metadata["source"] = paper_dict[id]
        graph_doc.source.metadata["id"] = str(id)
        if not args.dev:
            pickle.dump(graph_doc, file)
    # except Exception as e:
    #     print(e)

if not args.dev:
    if args.target != "both":
        f.close()
    else:
        ppi_f.close()
        tf_f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
