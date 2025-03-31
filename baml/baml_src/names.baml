
class AltNames {
  alt_names string[] @description("A list containing alternative names for a protein")
}

function CreateAltNames(input: string) -> AltNames {
  client Llama38b // will be overruled by the client registry
  prompt #"
    Generate up to 10 alternative names for the following protein:

    {{ ctx.output_format(prefix="Use the following OUTPUT FORMAT:") }}

    {{ _.role("user") }}

    {{ input }}
  "#
}
