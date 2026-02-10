class CardiacDecision {
  answer "Yes" | "No" @description("Whether the paper is cardiovascular focused.")
}

function CardiacFilterQ1(text: string) -> CardiacDecision {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    You are an expert cardiologist and molecular cardio-biologist. Please verify whether this scientific paper is explicitly about heart function/disease, cardiovascular development, or vascular disease. Please only give me a Yes or No answer.

    {{ _.role("user") }}
    TEXT:
    {{ text }}

    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}
  "#
}

function CardiacFilterQ2(text: string) -> CardiacDecision {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    You are an expert cardiologist and molecular cardio-biologist. Please verify whether a major focus of this scientific paper is centered towards cardiovascular/heart function and outcomes. Please only give me a Yes or No answer.

    {{ _.role("user") }}
    TEXT:
    {{ text }}

    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}
  "#
}

function CardiacFilterQ3(text: string) -> CardiacDecision {
  client Llama38b // will be overruled by the client registry
  prompt #"
    {{ _.role("system") }}
    You are an expert cardiologist and molecular cardio-biologist. Please verify whether this scientific paper includes most of its content or at least a major section dedicated to heart/cardiovascular domain. Please only give me a Yes or No answer.

    {{ _.role("user") }}
    TEXT:
    {{ text }}

    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}
  "#
}
