
class Triple {
  head string @description("head entity of the triple ")
  relation "INTERACTS_WITH" @description("relationship type")
  tail string @description("tail entity protein name of the triple")
}

class Triples {
  triples Triple[] @description("list of triples that describe interactions between two biological entities")
}

class Entities {
  entities string[] @description("list of biological entities")
}

function ExtractRelationships(sys_prompt: string, prompt: string, text: string) -> Triples {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    {{ sys_prompt }}
    {{ ctx.output_format(prefix="Use the following output format:") }}
    {{ _.role("user") }}
    {{ prompt }}
    Text: {{ text }}
  "#
}

class Message {
  role "user" | "assistant"
  content string
}

function ChatExtractRelationships(sys_prompt: string, init_prompt: string, text: string, messages: Message[]) -> Triples {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    {{ sys_prompt }}
    {{ _.role("user") }}
    {{ init_prompt }}
    {{ ctx.output_format(prefix="Use the following output format:") }}
    Text: {{ text }}
    {% for message in messages %}
      {{ _.role(message.role) }} 
      {{ message.content }}
    {% endfor %}
  "#
}

function GeneralChatExtractRelationships(sys_prompt: string, text: string, messages: Message[]) -> Triples {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    {{ sys_prompt }}
    {{ _.role("user") }}
    TEXT: {{ text }}
    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}
    USER QUESTION:
    {% for message in messages %}
      {{ _.role(message.role) }} 
      {{ message.content }}
    {% endfor %}
  "#
}

function ExtractNEs(sys_prompt: string, text: string, message: Message) -> Entities {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    {{ sys_prompt }}
    {{ _.role("user") }}
    Text: {{ text }}
    {{ ctx.output_format(prefix="Use the following output format:") }}
    USER QUESTION:
    {{ _.role(message.role) }} 
    {{ message.content }}
  "#
}
