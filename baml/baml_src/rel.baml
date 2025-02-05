
class Triple {
  head string @description("head protein name of the triple ")
  relation "INTERACTS_WITH" @description("relationship type")
  tail string @description("tail protein name of the triple")
}

class Triples {
  //   triples string[] @description("list of triples with protein-protein interactions")
  triples Triple[] @description("list of triples with protein-protein interactions")
}

function ExtractRelationshipTriples(input: string) -> Triples {
//   client Deepseek8b
  client Llama38b
  prompt #"
    Extract from the following INPUT all protein-protein interactions.
    {{ ctx.output_format }}
    {{ _.role("user") }}
    INPUT: {{ input }}
  "#
}
