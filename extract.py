import ast
import json
import pickle
from parser import args

from const import PROMPT_LOOKUP
from dicts import example_dict, ner_example_dict, schema_dict
from get_documents import all_ner_paths, documents, true_ner_paths, whole_documents
from graph_utils import build_graphdoc
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
from parsing import parse_msg2triples, parse_ners
from paths import (
    graphdoc_pkl_path,
    ner_json_path,
    ppi_graphdoc_pkl_path,
    tf_graphdoc_pkl_path,
)
from structured_classes import GenesAndTranscriptionFactors, Proteins, Triples
from style_templates import style_dict
from templates import (
    ProteinIndividualAllNersTemplate,
    ProteinIndividualTrueNersTemplate,
    ProteinNerTemplate,
    TfGeneIndividualAllNersTemplate,
    TfGeneIndividualTrueNersTemplate,
    TfGeneNerTemplate,
    TripleTemplate,
)

NER_SWITCH = False

init_string = style_dict[args.style][args.mode][PROMPT_LOOKUP][0]

triple_chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=init_string),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


if PROMPT_LOOKUP == "ppi":
    triple_template = (
        ProteinIndividualAllNersTemplate
        if args.nerrel == "individual" and not args.relgiventrueners
        else (
            ProteinIndividualTrueNersTemplate
            if args.relgiventrueners
            else TripleTemplate
        )
    )
elif PROMPT_LOOKUP == "tf":
    triple_template = (
        TfGeneIndividualAllNersTemplate
        if args.nerrel == "individual" and not args.relgiventrueners
        else (
            TfGeneIndividualTrueNersTemplate
            if args.relgiventrueners
            else TripleTemplate
        )
    )

triple_parser = JsonOutputParser(pydantic_object=Triples)

init_triple_prompt = PromptTemplate(
    template=triple_template,
    input_variables=["input"],
    partial_variables={
        "format_instructions": triple_parser.get_format_instructions(),
        "node_labels": False,
        "rel_types": ["INTERACTS_WITH"],
        "examples": example_dict[PROMPT_LOOKUP] if not args.noexamples else False,
    },
)

if args.nerrel:
    if PROMPT_LOOKUP == "ppi":
        ner_parser = JsonOutputParser(pydantic_object=Proteins)
    elif PROMPT_LOOKUP == "tf":
        ner_parser = JsonOutputParser(pydantic_object=GenesAndTranscriptionFactors)
    ner_template = ProteinNerTemplate if PROMPT_LOOKUP == "ppi" else TfGeneNerTemplate
    init_ner_prompt = PromptTemplate(
        template=ner_template,
        input_variables=["input"],
        partial_variables={
            "format_instructions": ner_parser.get_format_instructions(),
            "examples": ner_example_dict[PROMPT_LOOKUP],
        },
    )

if not args.dev:
    if args.target != "both":
        if args.startfromdoc == 0:
            f = open(graphdoc_pkl_path, "wb")
        else:
            f = open(graphdoc_pkl_path, "ab+")
        print(f"Open file {graphdoc_pkl_path}")
        if args.nerrel:
            if args.startfromdoc == 0:
                ner_f = open(ner_json_path, "w")
            else:
                ner_f = open(ner_json_path, "a+")
            print(f"Open file {ner_json_path}")
    else:
        if args.startfromdoc == 0:
            ppi_f = open(ppi_graphdoc_pkl_path, "wb")
            tf_f = open(tf_graphdoc_pkl_path, "wb")
        else:
            ppi_f = open(ppi_graphdoc_pkl_path, "ab+")
            tf_f = open(tf_graphdoc_pkl_path, "ab+")
        print(f"Open files {ppi_graphdoc_pkl_path, tf_graphdoc_pkl_path}")

interact_llm = llm.with_structured_output(Triples, include_raw=True)
graph_chain = triple_chat_prompt | interact_llm
if args.nerrel:
    if PROMPT_LOOKUP == "ppi":
        ner_llm = llm.with_structured_output(Proteins, include_raw=True)
    elif PROMPT_LOOKUP == "tf":
        ner_llm = llm.with_structured_output(
            GenesAndTranscriptionFactors, include_raw=True
        )
    ner_chain = triple_chat_prompt | ner_llm

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    if NER_SWITCH:
        response = ner_chain.invoke({"messages": state["messages"]})
    else:
        response = graph_chain.invoke({"messages": state["messages"]})
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


target_docs = documents if args.level == "chunks" else whole_documents
print(len(target_docs))


def query(app, doc, id):
    global NER_SWITCH
    ners, final_message, ppi_final_message, tf_final_message, prev_msgs, msg = (
        None,
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
            NER_SWITCH = True
            msg = app.invoke(msg_dict, config)
            ner_answer = msg["messages"][-1]
            ners = parse_ners(ner_answer)
            NER_SWITCH = False
            if not args.dev:
                ner_obj = repair_json(ners, return_objects=True)
                if ner_obj:
                    ner_obj = ner_obj["entities"]
                    if isinstance(ner_obj, str):
                        try:
                            ner_obj = ast.literal_eval(ner_obj)
                        except SyntaxError:
                            pass
                if args.filelist:
                    if not isinstance(ner_obj, str):
                        ner_obj.append(str(doc.metadata["file_path"]))
                    else:
                        ner_obj = [str(doc.metadata["file_path"])]
                json.dump(
                    ner_obj,
                    ner_f,
                    indent=4,
                )
            if args.onlyner:
                return (
                    ners,
                    final_message,
                    ppi_final_message,
                    tf_final_message,
                    prev_msgs,
                    msg,
                )
        elif args.nerrel == "individual":
            if not (args.relgiventrueners or args.relgivenallners):
                ner_answer = ner_chain.invoke(msg_dict, config)
                if ner_answer["parsed"]:
                    ners = ner_answer["parsed"].model_dump()
                elif "tool_calls" in ner_answer["raw"].additional_kwargs:
                    ners = ner_answer["raw"].additional_kwargs["tool_calls"][0][
                        "function"
                    ]["arguments"]
                elif "tool_calls" in ner_answer["raw"].response_metadata["message"]:
                    ners = ner_answer["raw"].response_metadata["message"]["tool_calls"][
                        0
                    ]["function"]["arguments"]
                ners = str(ners)
            elif args.relgiventrueners:
                try:
                    ner_path = [
                        x
                        for x in true_ner_paths
                        if doc.metadata["file_path"].stem == x.stem
                    ][0]
                    ners = open(ner_path, "r").readlines()
                    ners = [x.strip() for x in ners]
                except:
                    ners = list()
            elif args.relgivenallners:
                ner_path = [
                    x for x in all_ner_paths if doc.metadata["file_path"].stem == x.stem
                ][0]
                ners = open(ner_path, "r").readlines()
                ners = [x.strip() for x in ners]
            if not args.dev:
                if not (args.relgivenallners or args.relgiventrueners):
                    ner_obj = repair_json(str(ners), return_objects=True)
                    if ner_obj:
                        ner_obj = ner_obj["entities"]
                        if isinstance(ner_obj, str):
                            try:
                                ner_obj = ast.literal_eval(ner_obj)
                            except SyntaxError:
                                pass
                    if args.filelist:
                        if not isinstance(ner_obj, str):
                            ner_obj.append(str(doc.metadata["file_path"]))
                        else:
                            ner_obj = [str(doc.metadata["file_path"])]
                    json.dump(
                        ner_obj,
                        ner_f,
                        indent=4,
                    )
            if args.onlyner:
                return (
                    ners,
                    final_message,
                    ppi_final_message,
                    tf_final_message,
                    prev_msgs,
                    msg,
                )
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
            for human_msg in style_dict[args.style][args.mode][PROMPT_LOOKUP][1:]:
                msg = app.invoke(
                    {"messages": [HumanMessage(human_msg)]},
                    config,
                )
            final_message = msg["messages"][-1]
        else:
            prev_msgs = list()
            for human_msg in style_dict[args.style][args.mode][PROMPT_LOOKUP][1:]:
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
                    HumanMessage(style_dict[args.style][args.mode][PROMPT_LOOKUP][1])
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
                    HumanMessage(style_dict[args.style][args.mode][PROMPT_LOOKUP][2])
                ]
            },
            config,
        )
        if args.style == 2:
            tf_final_message = msg["messages"][-1]
        elif args.style == 3:
            ppi_final_message = msg["messages"][-1]
    return ners, final_message, ppi_final_message, tf_final_message, prev_msgs, msg


for i, (doc, id) in enumerate(
    target_docs[
        args.startfromdoc : len(target_docs) if args.untildoc == 0 else args.untildoc
    ],
    args.startfromdoc,
):
    # try:
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # result = attempt(
    #     tries=1,
    #     seconds=int(1e6),
    #     func=query,
    #     kwargs={"app": app, "doc": doc, "id": id},
    # )
    # try:
    # except TypeError as e:
    #     continue
    results = query(app=app, doc=doc, id=id)
    ners, final_message, ppi_final_message, tf_final_message, prev_msgs, msg = results

    if args.onlyner:
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

    for fm, file in zip(final_messages, files):
        triples = parse_msg2triples(fm)
        if args.style == 6:
            for prev_msg in prev_msgs:
                prev_triples = parse_msg2triples(prev_msg)
                triples += prev_triples
        print(i, triples)
        graph_doc = build_graphdoc(triples, doc, id)
        if args.saveinbetweenoutputs:
            previous_graphdocs = list()
            for i in range(
                3 if args.nerrel == "conversational" else 1,
                len(msg["messages"]) - 2,
                2,
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
