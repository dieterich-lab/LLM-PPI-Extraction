from parser import args

import hnswlib
import numpy as np
from ollama import Client
from tqdm import tqdm

from dataset import get_dataset

ip_dict = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
    "mk22d": "10.250.135.115",
}

dim = 384
index_path = "/prj/LINDA_LLM/outputs/regulatome_train_idx.bin"

embed_model = "all-minilm"
port_dict = {"g2": 32, "g3": 33, "g4": 34, "g5": 35}
port = port_dict[args.node]

client = Client(
    host=f"http://{ip_dict[args.node]}:114{port}",
)


def save_index():
    train_dataset, dev_dataset, _ = get_dataset()

    # model = "mxbai-embed-large"

    embeds = list()
    for sample in tqdm(train_dataset):
        embed = client.embed(
            model=embed_model,
            input=sample["doc"],
        ).embeddings
        embeds.append(embed[0])

    for sample in tqdm(dev_dataset):
        embed = client.embed(
            model=embed_model,
            input=sample["doc"],
        ).embeddings
        embeds.append(embed[0])

    embeds = np.array(embeds)

    p = hnswlib.Index(space="l2", dim=dim)

    p.init_index(max_elements=len(embeds), ef_construction=100, M=dim)
    p.set_ef(10)
    p.set_num_threads(4)
    p.add_items(embeds)

    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)


def load_index():
    p = hnswlib.Index(space="l2", dim=dim)
    p.load_index(index_path)
    return p


if __name__ == "__main__":
    save_index()
