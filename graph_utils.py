from get_documents import paper_dict
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


def build_graphdoc(triples, doc, id):
    nodes_set = set()
    rels = list()
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
    graph_doc.source.metadata["source"] = paper_dict[id]
    graph_doc.source.metadata["id"] = str(id)
    return graph_doc
