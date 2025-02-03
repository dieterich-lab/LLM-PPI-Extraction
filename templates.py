TripleTemplate = """Based on the following example, extract entities and 
relations from the provided text.

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES:

{rel_types}

Below are a number of examples of text and their extracted entities and relationships.

### Examples:

{examples}

Use the following JSON format:

### Format Instructions:

{format_instructions}

### Text: 

{input}
"""

ProteinIndividualAllNersTemplate = """In the following you'll find a list of 
PROTEINS  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only of those PROTEINS, that interact with each other, based on the text.

IMPORTANT: Read the TEXT carefully. Not all PROTEINS are candidates for interactions.

### PROTEINS

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Below are a number of examples of text and their extracted entities and relationships.

### EXAMPLES

{examples}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

ProteinIndividualTrueNersTemplate = """In the following you'll find a list of 
PROTEINS  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only of those PROTEINS, that interact with each other, based on the text.

### PROTEINS

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Below are a number of examples of text and their extracted entities and relationships.

### EXAMPLES

{examples}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

ProteinIndividualNoExamplesTemplate = """In the following you'll find a list of 
PROTEINS  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only of those PROTEINS, that interact with each other, based on the text.

IMPORTANT: Read the TEXT carefully. Not all PROTEINS are candidates for interactions.

### PROTEINS

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

TfGeneIndividualAllNersTemplate = """In the following you'll find a list of 
TRANSCRIPTION FACTORS and GENES  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only beteween those TRANSCRIPTION FACTORS and GENES that interact with each other, based on the text.

IMPORTANT: Read the TEXT carefull. Not all TRANSCRIPTION FACTORS and GENES are candidates for interactions.

### TRANSCRIPTION FACTORS and GENES

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Below are a number of examples of text and their extracted entities and relationships.

### EXAMPLES

{examples}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

TfGeneIndividualTrueNersTemplate = """In the following you'll find a list of 
TRANSCRIPTION FACTORS and GENES  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only beteween those TRANSCRIPTION FACTORS and GENES that interact with each other, based on the text.

### TRANSCRIPTION FACTORS and GENES

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Below are a number of examples of text and their extracted entities and relationships.

### EXAMPLES

{examples}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

TfGenesIndividualNoExamplesTemplate = """In the following you'll find a list of 
TRANSCRIPTION FACTORS and GENES  that have been extracted from the provided TEXT that is listed at the end of this prompt.
You are now to extract relations only beteween those TRANSCRIPTION FACTORS and GENES that interact with each other, based on the text.

IMPORTANT: Read the TEXT carefull. Not all TRANSCRIPTION FACTORS and GENES are candidates for interactions.

### TRANSCRIPTION FACTORS and GENES

{entities}

Use the following relation types, don't use any other relation that is not defined below:

### RELATION TYPES

{rel_types}

Be aware that the relationships signal also the direction of the participants. Use the following JSON format:

### FORMAT INSTRUCTIONS

{format_instructions}

### TEXT: 

{input}
"""

ProteinNerTemplate = """Based on the following example, extract all proteins
from the provided text.

Below are a number of examples of text passages and corresponding extracted proteins.

### Examples:

{examples}

Use and adhere to the following JSON format:

### Format instructions:

{format_instructions}

### Text: 

{input}
"""

TfGeneNerTemplate = """Based on the following example, extract all genes and transcription factors
from the provided text.

Below are a number of examples of text passages and corresponding extracted genes and transcription factors.

### Examples:

{examples}

Use and adhere to the following JSON format:

### Format instructions:

{format_instructions}

### Text: 

{input}
"""
