from parser import args

from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama

model_dict = {
    "8b": "llama3.1:8b",
    "70b": "llama3.1:70b",
    "405b": "llama3.1:405b",
    "mixtral": "mixtral:8x22b",
    "biollm": "taozhiyuai/openbiollm-llama-3:70b_q4_k_m",
    "nemo": "mistral-nemo",
}
model = model_dict[args.model]

ip_dict = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
}
# llm = OllamaLLM(

if not args.style:
    llm = Ollama(
        model=model,
        temperature=0,
        keep_alive="24h",
        base_url=f"http://{ip_dict[args.gpu]}:114{args.port}",
    )
else:
    llm = ChatOllama(
        model=model,
        temperature=0,
        keep_alive="24h",
        base_url=f"http://{ip_dict[args.gpu]}:114{args.port}",
    )
