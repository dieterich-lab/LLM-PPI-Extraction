import json
import pickle
from parser import args

from const import PROMPT_LOOKUP
from get_documents import documents, whole_documents
from graph_utils import attempt, build_graphdoc, parse_msg2triples
from json_repair import repair_json
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
from paths import (
    graphdoc_pkl_path,
    ner_json_path,
    ppi_graphdoc_pkl_path,
    tf_graphdoc_pkl_path,
)
from structured_classes import (
    GenesAndTranscriptionFactors,
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
    LR_EXAMPLES_SIMPLE,
    LR_INTERACTIONS,
    LR_NODE_LABELS,
    PPI_EXAMPLES,
    PPI_EXAMPLES_SIMPLE,
    PPI_INTERACTIONS,
    PPI_NER_CONVERSATIONAL_TEMPLATE,
    PPI_NER_EXAMPLES_SIMPLE,
    PPI_NER_INDIVIDUAL_TEMPLATE,
    PPI_NODE_LABELS,
    TF_EXAMPLES,
    TF_EXAMPLES_SIMPLE,
    TF_INTERACTIONS,
    TF_NER_CONVERSATIONAL_TEMPLATE,
    TF_NER_EXAMPLES_SIMPLE,
    TF_NER_INDIVIDUALTEMPLATE,
    TF_NODE_LABELS,
    TRIPLE_TEMPLATE,
    TRIPLE_TEMPLATE_SIMPLE,
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

mode = (
    f"nerrel_{args.nerrel}" if args.nerrel else "simple" if args.simple else "complex"
)
triple_basestring_parts = style_dict[args.style][mode][PROMPT_LOOKUP][0]

triple_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=triple_basestring_parts),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


triple_schema = (
    schema_dict[PROMPT_LOOKUP] if not args.simple else simple_schema_dict[PROMPT_LOOKUP]
)


triple_template = (
    PPI_NER_INDIVIDUAL_TEMPLATE
    if args.nerrel == "individual" and args.target == "ppi"
    else (
        TF_NER_INDIVIDUALTEMPLATE
        if args.nerrel == "individual" and args.target == "tf"
        else TRIPLE_TEMPLATE if not args.simple else TRIPLE_TEMPLATE_SIMPLE
    )
)

triple_parser = JsonOutputParser(pydantic_object=triple_schema)

init_triple_prompt = PromptTemplate(
    template=triple_template,
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

if args.nerrel:
    kw = "proteins" if PROMPT_LOOKUP == "ppi" else "genes_and_transcriptionfactors"
    if PROMPT_LOOKUP == "ppi":
        ner_parser = JsonOutputParser(pydantic_object=Proteins)
    elif PROMPT_LOOKUP == "tf":
        ner_parser = JsonOutputParser(pydantic_object=GenesAndTranscriptionFactors)
    init_ner_prompt = PromptTemplate(
        template=(
            PPI_NER_CONVERSATIONAL_TEMPLATE
            if args.target == "ppi"
            else TF_NER_CONVERSATIONAL_TEMPLATE
        ),
        input_variables=["input"],
        partial_variables={
            "format_instructions": ner_parser.get_format_instructions(),
            "examples": ner_example_dict[PROMPT_LOOKUP],
        },
    )

f = None
ppi_f = None
tf_f = None
ner_f = None
if not args.dev:
    if args.target != "both":
        f = open(graphdoc_pkl_path, "wb")
        if args.nerrel:
            ner_f = open(ner_json_path, "w")
    else:
        ppi_f = open(ppi_graphdoc_pkl_path, "wb")
        tf_f = open(tf_graphdoc_pkl_path, "wb")

if args.style:
    interact_llm = llm.with_structured_output(triple_schema, include_raw=True)
    graph_chain = triple_prompt | interact_llm
    if args.nerrel:
        if PROMPT_LOOKUP == "ppi":
            ner_llm = llm.with_structured_output(Proteins, include_raw=True)
        elif PROMPT_LOOKUP == "tf":
            ner_llm = llm.with_structured_output(
                GenesAndTranscriptionFactors, include_raw=True
            )
        ner_chain = triple_prompt | ner_llm
        # ner_chain = ner_prompt | ner_llm
        # if args.nerrel == "conversational":
        #     ner_chain = triple_prompt | ner_llm
        # elif args.nerrel == "individual":
        #     ner_chain = ner_prompt | ner_llm

    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        if NER:
            response = ner_chain.invoke({"messages": state["messages"]})
        else:
            response = graph_chain.invoke({"messages": state["messages"]})
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)


target_docs = documents if not args.doclevel else whole_documents
print(len(target_docs))


def query(app, doc, id):
    global NER
    ners, ppi_final_message, tf_final_message, prev_msgs, msg = (
        None,
        None,
        None,
        None,
        None,
    )
    config = {"configurable": {"thread_id": id}}
    if not args.nerrel:
        msg = app.invoke(
            {
                "messages": [
                    HumanMessage(init_triple_prompt.format(input=doc.page_content))
                ]
            },
            config,
        )
    else:  # nerrel
        msg_dict = {
            "messages": [HumanMessage(init_ner_prompt.format(input=doc.page_content))]
        }
        if args.nerrel == "conversational":
            NER = True
            msg = app.invoke(msg_dict, config)
            ner_answer = msg["messages"][-1]
            if "parsed" in ner_answer and ner_answer["parsed"]:
                ners = str(ner_answer["parsed"].model_dump())
            elif "tool_calls" in ner_answer.additional_kwargs and kw in repair_json(
                ner_answer.additional_kwargs["tool_calls"][0]["function"]["arguments"],
                return_objects=True,
            ):
                ners = repair_json(
                    ner_answer.additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            elif (
                "raw" in ner_answer
                and "tool_calls" in ner_answer["raw"].additional_kwargs
            ):
                ners = str(
                    ner_answer["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            elif "raw" in ner_answer and (
                "tool_calls" in ner_answer["raw"].response_metadata["message"]
            ):
                ners = str(
                    ner_answer["raw"].response_metadata["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                )
            elif (
                "tool_calls" in ner_answer.response_metadata["message"]
                and ner_answer.response_metadata["message"]["tool_calls"][0][
                    "function"
                ]["arguments"]
            ):
                ners = str(
                    ner_answer.response_metadata["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                )
            NER = False
            if not args.dev:
                obj = repair_json(ners, return_objects=True)
                if obj:
                    obj = obj[kw]
                json.dump(
                    obj,
                    ner_f,
                    indent=4,
                )
            if args.onlyner:
                return ners, ppi_final_message, tf_final_message, prev_msgs, msg
        elif args.nerrel == "individual":
            ner_answer = ner_chain.invoke(msg_dict, config)
            if ner_answer["parsed"]:
                ners = str(ner_answer["parsed"].model_dump())
            elif "tool_calls" in ner_answer["raw"].additional_kwargs:
                ners = ner_answer["raw"].additional_kwargs["tool_calls"][0]["function"][
                    "arguments"
                ]
            elif "tool_calls" in ner_answer["raw"].response_metadata["message"]:
                ners = ner_answer["raw"].response_metadata["message"]["tool_calls"][0][
                    "function"
                ]["arguments"]
            if not args.dev:
                json.dump(
                    repair_json(ners, return_objects=True)[kw],
                    ner_f,
                    indent=4,
                )
            if args.onlyner:
                return ners, ppi_final_message, tf_final_message, prev_msgs, msg
            msg = app.invoke(
                {
                    "messages": [
                        HumanMessage(
                            init_triple_prompt.format(
                                input=doc.page_content, entities=ners
                            )
                        )
                    ]
                },
                config,
            )

    if args.target != "both":
        if args.style != 6:
            for human_msg in style_dict[args.style][mode][PROMPT_LOOKUP][1:]:
                msg = app.invoke(
                    {"messages": [HumanMessage(human_msg)]},
                    config,
                )
            final_message = msg["messages"][-1]
        else:
            prev_msgs = list()
            for human_msg in style_dict[args.style][mode][PROMPT_LOOKUP][1:]:
                msg = app.invoke(
                    {"messages": [HumanMessage(human_msg)]},
                    config,
                )
                prev_msgs.append(msg["messages"][-1])
            prev_msgs.pop(-1)
            final_message = msg["messages"][-1]
    else:
        msg = app.invoke(
            {
                "messages": [
                    HumanMessage(style_dict[args.style][mode][PROMPT_LOOKUP][1])
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
                    HumanMessage(style_dict[args.style][mode][PROMPT_LOOKUP][2])
                ]
            },
            config,
        )
        if args.style == 2:
            tf_final_message = msg["messages"][-1]
        elif args.style == 3:
            ppi_final_message = msg["messages"][-1]
    return ners, ppi_final_message, tf_final_message, prev_msgs, msg


for i, (doc, id) in enumerate(
    target_docs[
        args.startfromdoc : len(target_docs) if args.untildoc == 0 else args.untildoc
    ],
    args.startfromdoc,
):
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    ners, ppi_final_message, tf_final_message, prev_msgs, msg = attempt(
        5, 300, query, kwargs={"app": app, "doc": doc, "id": id}
    )

    if args.onlyner:
        print(i, ners)
        continue
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
        if args.style == 6:
            for prev_msg in prev_msgs:
                prev_triples = parse_msg2triples(prev_msg)
                triples += prev_triples
        print(i, triples)
        graph_doc = build_graphdoc(triples, doc, id)
        if args.saveinbetweenoutputs:
            previous_graphdocs = list()
            for i in range(
                3 if args.nerrel == "conversational" else 1, len(msg["messages"]) - 2, 2
            ):
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
        if args.nerrel:
            ner_f.close()
    else:
        ppi_f.close()
        tf_f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
if args.nerrel:
    print(f"Finished writing NERs to {ner_json_path}.")
