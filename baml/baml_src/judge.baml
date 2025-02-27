
class Reason {
  reason string @description("a concise scientific one-sentence that reason about a question")
}

function Judge(sys_prompt: string, text: string, reasoning: string, triple: string, prompt: string) -> Reason {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    {{ sys_prompt }}
    TEXT: {{ text }}
    TRIPLE: {{ triple }}
    REASONING: {{ reasoning }}
    USER QUESTION:
    {{ _.role("user") }}
    {{ prompt }}
    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}
  "#
}
