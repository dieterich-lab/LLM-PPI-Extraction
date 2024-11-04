import argparse
import json
import os
import pickle
from pathlib import Path

from graph_utils import escape_json
from json_repair import repair_json
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langchain_text_splitters import MarkdownTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from structured_classes import (
    PPI_Triples,
    PPI_Triples_Simple,
    TF_Triples,
    TF_Triples_Simple,
)
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    nargs="?",
    const="tf",
    type=str,
    default="ppi",
    choices=["tf", "ppi", "both"],
)
parser.add_argument(
    "--parser",
    nargs="?",
    const="llama_parse",
    type=str,
    default="llama_parse",
    choices=["llama_parse", "marker"],
)
parser.add_argument(
    "--simple",
    action="store_true",
)
parser.add_argument(
    "--curated",
    action="store_true",
)
parser.add_argument(
    "--dev",
    action="store_true",
)
parser.add_argument(
    "--model", choices=["8b", "70b", "405b", "mixtral", "biollm"], default="8b"
)
args = parser.parse_args()

old = True

ending_dict = {"marker": "md", "llama_parse": "txt"}
basestring_dict = {"ppi": PPI_BASESTRINGPARTS, "tf": TF_BASESTRINGPARTS}
simple_basestring_dict = {
    "ppi": PPI_BASESTRINGPARTS_SIMPLE,
    "tf": TF_BASESTRINGPARTS_SIMPLE,
}
example_dict = {"ppi": PPI_EXAMPLES, "tf": TF_EXAMPLES}
simple_example_dict = {"ppi": PPI_EXAMPLES_SIMPLE, "tf": TF_EXAMPLES_SIMPLE}
interactions_dict = {"ppi": PPI_INTERACTIONS, "tf": TF_INTERACTIONS}
nodelabels_dict = {"ppi": PPI_NODE_LABELS, "tf": TF_NODE_LABELS}
schema_dict = {"ppi": PPI_Triples, "tf": TF_Triples}
simple_schema_dict = {"ppi": PPI_Triples_Simple, "tf": TF_Triples_Simple}


model_dict = {
    "8b": "llama3.1:8b",
    "70b": "llama3.1:70b",
    "405b": "llama3.1:405b",
    "mixtral": "mixtral:8x22b",
    "biollm": "taozhiyuai/openbiollm-llama-3:70b_q4_k_m",
}
model = model_dict[args.model]
g4 = "10.250.135.153"
g2 = "10.250.135.143"
g3 = "10.250.135.150"
g5 = "10.250.135.156"
sg500 = "10.250.135.128"
port34 = 11434
port35 = 11435
port36 = 11436
llm = ChatOllama(
    model=model, temperature=0, keep_alive="24h", base_url=f"http://{g4}:{port35}"
)

text_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
paper_dict = dict()

if old and args.task in ["tf", "both"]:
    _task = "ppi"
else:
    _task = args.task

_paper_paths = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/{_task}/{args.parser}"
)
if args.curated:
    _paper_paths = _paper_paths / "5curated"
else:
    _paper_paths = _paper_paths / "100samples"

paper_paths = list(_paper_paths.glob(f"*.{ending_dict[args.parser]}"))


paper_pkl_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_chunks/{_task}/{args.parser}/paper_chunks.pkl"
)
if args.curated:
    paper_pkl_path = paper_pkl_path.parent / "5curated" / "paper_chunks.pkl"
else:
    paper_pkl_path = paper_pkl_path.parent / "100samples" / "paper_chunks.pkl"

paper_dict_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_dicts/{_task}/{args.parser}/paper_dict.pkl"
)
if args.curated:
    paper_dict_path = paper_dict_path.parent / "5curated" / "paper_chunks.pkl"
else:
    paper_dict_path = paper_dict_path.parent / "100samples" / "paper_chunks.pkl"

os.makedirs(Path(paper_pkl_path).parent, exist_ok=True)
f = open(paper_pkl_path, "wb")
for i, x in enumerate(paper_paths):
    print(i, x)
    paper_dict[i] = str(x)
    text = open(x, "r").read().strip()
    if text:
        texts = text_splitter.create_documents([text])
        for t in texts:
            pickle.dump((t, i), f)
f.close()
os.makedirs(Path(paper_dict_path).parent, exist_ok=True)
with open(paper_dict_path, "w") as f:
    json.dump(paper_dict, f, indent=4)

with open(paper_dict_path, "r") as f:
    paper_dict = json.load(f)
    paper_dict = {int(k): v for k, v in paper_dict.items()}

documents = list()
with open(paper_pkl_path, "rb") as f:
    while 1:
        try:
            documents.append(pickle.load(f))
        except EOFError:
            break
print(len(documents))


schema = schema_dict[args.task] if not args.simple else simple_schema_dict[args.task]
graph_schema = schema
graph_schema = convert_to_openai_tool(schema)
triple_len = len(
    graph_schema["function"]["parameters"]["properties"]["triples"]["items"][
        "properties"
    ]
)

pydantic_parser = PydanticOutputParser(pydantic_object=schema)
system_prompt = "\n".join(
    basestring_dict[args.task] if not args.simple else simple_basestring_dict[args.task]
)

human_prompt = (
    TEMPLATE.format(
        node_labels=nodelabels_dict[args.task],
        rel_types=interactions_dict[args.task],
        examples=escape_json(json.dumps(example_dict[args.task])),
        format_instructions=escape_json(pydantic_parser.get_format_instructions()),
        input="{input}",
    )
    if not args.simple
    else TEMPLATE_SIMPLE.format(
        rel_types=interactions_dict[args.task],
        examples=escape_json(json.dumps(example_dict[args.task])),
        input="{input}",
    )
)

graph_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

conv_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


graph_llm = llm.with_structured_output(graph_schema, include_raw=True)
graph_chain = graph_prompt | graph_llm

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = graph_llm.invoke(state["messages"])
    # Update message history with response:
    return {"messages": response}


config = {"configurable": {"thread_id": "abc123"}}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if old and args.task == "tf":
    graph_doc_filename = "graph_documents_old.pkl"
else:
    graph_doc_filename = "graph_documents.pkl"


graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{args.task}/{args.parser}/{model}/graph_documents.pkl"

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )

if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename
os.makedirs(Path(graphdoc_pkl_path).parent, exist_ok=True)

graph_documents = list()

if not args.dev:
    f = open(graphdoc_pkl_path, "wb")

for i, (doc, id) in enumerate(documents):
    print(i)
    c = 0
    try:
        # while c < 5:
        #     try:
        #         with Timeout(60):
        #             msg = graph_chain.invoke(
        #                 {
        #                     "input": doc.page_content,
        #                 }
        #             )
        #             break
        #     except Timeout.Timeout:
        #         print("Timeout")
        #         c += 1
        # break
        # input_messages = [HumanMessage(doc.page_content)]
        # output = app.invoke({"messages": input_messages}, config)
        msg = graph_chain.invoke(
            {
                "input": doc.page_content,
            }
        )
        if msg is None:
            parsed = []
        else:
            parsed = msg["parsed"]
            if parsed is None:
                _parsed = repair_json(msg["raw"].content, return_objects=True)
                for p in _parsed:
                    if not isinstance(p, dict):
                        continue
                    parsed = [t for t in p["triples"] if len(t) >= triple_len]
                    break
            else:
                parsed = repair_json(msg["parsed"]["triples"], return_objects=True)

        if not parsed:
            continue
        nodes_set = set()
        rels = list()
        for triple in parsed:
            n1 = triple[0]
            n2 = triple[2]
            nodes_set.add(n1)
            nodes_set.add(n2)
            rels.append(
                Relationship(source=Node(id=n1), target=Node(id=n2), type=triple[1])
            )
        nodes = [Node(id=el) for el in list(nodes_set)]
        graph_doc = GraphDocument(nodes=nodes, relationships=rels, source=doc)
        pickle.dump(graph_doc, f)
    except Exception as e:
        print(e)

if not args.dev:
    f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")

"""
first={
  raw: RunnableBinding(bound=ChatOllama(model='mistral-nemo', temperature=0.0, keep_alive='24h', base_url='http://10.250.135.153:11435'), kwargs={'tools': [{'type': 'function', 'function': {'name': 'Triples', 'description': '', 'parameters': {'properties': {'triples': {'description': 'Liste aller extrahierten Triples', 'items': {'properties': {'head': {'description': 'Beschreibt die Start-Entität', 'type': 'string'}, 'relation': {'description': 'Beschreibt die Relation zwischen Start-Entität und Ziel-Entität. Darf ein beliebiger Freitext sein.', 'type': 'string'}, 'tail': {'description': 'Beschreibt die Ziel-Entität', 'type': 'string'}}, 'required': ['head', 'relation', 'tail'], 'type': 'object'}, 'type': 'array'}}, 'required': ['triples'], 'type': 'object'}}}], 'tool_choice': 'any'}, config={}, config_factories=[])
} middle=[] last=RunnableWithFallbacks(runnable=RunnableAssign(mapper={
  parsed: RunnableLambda(itemgetter('raw'))
          | JsonOutputKeyToolsParser(first_tool_only=True, key_name='Triples'),
  parsing_error: RunnableLambda(...)
}), fallbacks=[RunnableAssign(mapper={
  parsed: RunnableLambda(lambda _: None)
})], exception_key='parsing_error')
"""

"""
{'triples': [['Nationale VersorgungsLeitlinie Chronische Herzinsuffizienz', 'ist eine medizinische Leitlinie über', 'Herzinssufizienz'], ['Version 4.0', 'ist die Version von', 'Nationale VersorgungsLeitlinie Chronische Herzinsuffizienz'], ['Langfassung AWMF-Register-Nr. nvl-006', 'ist eine lange Fassung mit der Registernummer', 'Nationale VersorgungsLeitlinie Chronische Herzinsuffizienz'], ['Träger: Bundesärztekammer Kassenärztliche Bundesvereinigung Arbeitsgemeinschaft der Wissenschaftlichen Medizinischen Fachgesellschaften', 'sind die Träger von', 'Nationale VersorgungsLeitlinie Chronische Herzinsuffizienz'], ['© 2023 KBV AWMF', 'ist das Copyright von', 'Nationale VersorgungsLeitlinie Chronische Herzinsuffizienz']]}
"""
