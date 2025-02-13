import json
import pickle


def convert_and_save_to_json(triple_pkl_path, triple_json_path):
    json_triples = list()
    with open(triple_pkl_path, "rb") as triple_pkl_file:
        while 1:
            try:
                tuple = pickle.load(triple_pkl_file)
                triple_objs = tuple[0]
                text = tuple[1]
                file_name = str(tuple[2])
                cur_triples = list()
                for triple_obj in triple_objs.triples:
                    cur_triples.append(
                        {
                            "head": triple_obj.head,
                            "relation": triple_obj.relation,
                            "tail": triple_obj.tail,
                        }
                    )
                json_triples.append(
                    {"triples": cur_triples, "text": text, "filename": file_name}
                )
            except EOFError:
                break

    with open(triple_json_path, "w") as triple_json_file:
        json.dump(json_triples, triple_json_file, indent=4)
