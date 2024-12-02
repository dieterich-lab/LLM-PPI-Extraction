import pickle
from parser import args

from const import PROMPT_LOOKUP
from get_documents import documents, whole_documents
from graph_utils import build_graphdoc, parse_msg2triples
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
    Proteins,
    TF_Triples,
    TF_Triples_Simple,
    Triples,
    Triples_Simple,
)
from style_templates import style_dict
from templates import (
    INTERACT_TEMPLATE,
    INTERACT_TEMPLATE_SIMPLE,
    LR_EXAMPLES_SIMPLE,
    LR_INTERACTIONS,
    LR_NODE_LABELS,
    NER_TEMPLATE_SIMPLE,
    PPI_EXAMPLES,
    PPI_EXAMPLES_SIMPLE,
    PPI_INTERACTIONS,
    PPI_NER_EXAMPLES_SIMPLE,
    PPI_NODE_LABELS,
    TF_EXAMPLES,
    TF_EXAMPLES_SIMPLE,
    TF_INTERACTIONS,
    TF_NER_EXAMPLES_SIMPLE,
    TF_NODE_LABELS,
)
from utils import Timeout

NER = False

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
ner_example_dict = {
    "ppi": PPI_NER_EXAMPLES_SIMPLE,
    "tf": TF_NER_EXAMPLES_SIMPLE,
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

mode = "nerrel" if args.nerrel else "simple" if args.simple else "complex"
basestring_parts = style_dict[args.style][mode][PROMPT_LOOKUP][0]

triple_schema = (
    schema_dict[PROMPT_LOOKUP] if not args.simple else simple_schema_dict[PROMPT_LOOKUP]
)


system_message = SystemMessage(content=basestring_parts)
base_prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="messages"),
    ]
)
triple_parser = JsonOutputParser(pydantic_object=triple_schema)
if args.nerrel:
    ner_parser = JsonOutputParser(pydantic_object=Proteins)
    init_ner_prompt = PromptTemplate(
        template=NER_TEMPLATE_SIMPLE,
        input_variables=["input"],
        partial_variables={
            "format_instructions": ner_parser.get_format_instructions(),
            "examples": ner_example_dict[args.target],
        },
    )

init_triple_prompt = PromptTemplate(
    template=INTERACT_TEMPLATE if not args.simple else INTERACT_TEMPLATE_SIMPLE,
    input_variables=["input"],
    partial_variables={
        "format_instructions": triple_parser.get_format_instructions(),
        "node_labels": False if args.simple else nodelabels_dict[PROMPT_LOOKUP],
        "rel_types": interactions_dict[PROMPT_LOOKUP],
        "examples": (
            example_dict[PROMPT_LOOKUP]
            if not args.simple
            else simple_example_dict[PROMPT_LOOKUP]
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
    interact_llm = llm.with_structured_output(triple_schema, include_raw=True)
    graph_runnable = triple_base_prompt | interact_llm
    # graph_runnable = base_prompt | interact_llm
    if args.nerrel:
        ner_llm = llm.with_structured_output(Proteins, include_raw=True)
        ner_runnable = base_ner_prompt | ner_llm
        # ner_runnable = base_prompt | ner_llm

    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        # if NER:
        #     response = ner_runnable.invoke({"messages": state["messages"]})
        # else:
        #     response = graph_runnable.invoke({"messages": state["messages"]})
        response = graph_runnable.invoke({"messages": state["messages"]})
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)


target_docs = documents if not args.doclevel else whole_documents
print(len(target_docs))

for i, (doc, id) in enumerate(
    target_docs[
        args.startfromdoc : len(target_docs) if args.untildoc == 0 else args.untildoc
    ],
    args.startfromdoc,
):
    c = 0
    # try:
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    while c < 5:
        try:
            with Timeout(1800):
                config = {"configurable": {"thread_id": id}}
                if not args.nerrel:
                    msg = app.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    init_triple_prompt.format(input=doc.page_content)
                                )
                            ]
                        },
                        config,
                    )
                else:
                    NER = True
                    msg = app.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    init_ner_prompt.format(input=doc.page_content)
                                )
                            ]
                        },
                        config,
                    )
                    NER = False
                if args.target != "both":
                    for human_msg in style_dict[args.style][mode][PROMPT_LOOKUP][1:]:
                        msg = app.invoke(
                            {"messages": [HumanMessage(human_msg)]},
                            config,
                        )
                    final_message = msg["messages"][-1]
                else:
                    msg = app.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    style_dict[args.style][mode][PROMPT_LOOKUP][1]
                                )
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
                                HumanMessage(
                                    style_dict[args.style][mode][PROMPT_LOOKUP][2]
                                )
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
        print(i, triples)
        graph_doc = build_graphdoc(triples, doc, id)
        if args.saveinbetweenoutputs:
            previous_graphdocs = list()
            for i in range(1, len(msg["messages"]) - 2, 2):
                previous_triples = parse_msg2triples(msg["messages"][i])
                previous_graphdocs.append(build_graphdoc(previous_triples, doc, id))
        if not args.dev:
            if not args.saveinbetweenoutputs:
                pickle.dump(graph_doc, file)
            else:
                pickle.dump([*previous_graphdocs, graph_doc], file)
    # except Exception as e:
    #     print(e)

if not args.dev:
    if args.target != "both":
        f.close()
    else:
        ppi_f.close()
        tf_f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
