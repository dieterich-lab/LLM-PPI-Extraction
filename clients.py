from parser import args

from baml_py import ClientRegistry

ip_dict = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
    "mk22d": "10.250.135.115",
}

ollama_client_names = [
    ("llama33", "llama3.3-128k:70b"),
    ("llama31", "llama3.1-128k:8b"),
    ("deepseek8b", "deepseek-r1-128k:8b"),
    ("deepseek70b", "deepseek-r1-128k:70b"),
    ("llama33regu", "hf.co/phiwi/Llama-3.3-70B-Instruct-regu"),
    ("llama31regu", "hf.co/phiwi/Meta-Llama-3.1-8B-regu"),
]

hf_client_names = {
    "llama33": "unsloth/Llama-3.3-70B-Instruct",
    "llama31": "unsloth/Meta-Llama-3.1-8B",
    "deepseek8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "llama33regu": "hf.co/phiwi/Llama-3.3-70B-Instruct-regu",
    "llama31regu": "hf.co/phiwi/Meta-Llama-3.1-8B-regu",
}


hf_model_id = hf_client_names[args.model]


clients = list()
for name, client in ollama_client_names:
    clients.append(
        {
            "name": name,
            "provider": "openai-generic",
            "options": {
                "base_url": f"http://{ip_dict[args.node]}:114{args.port}/v1",
                "model": client,
                "max_tokens": 10000,
                "temperature": 0.0,
                # "temperature": 0.6,  # from huggingface usage recommendations https://huggingface.co/deepseek-ai/DeepSeek-R1#usage-recommendations
                "n_ctx": 10,
            },
        }
    )

cr = ClientRegistry()
for client in clients:
    cr.add_llm_client(**client)
cr.set_primary(args.model)
