style_dict = {
    1: {
        "simple": {
            "ppi": [
                "You are a top-tier molecular biologist specialized in the field of "
                "cardiology and molecular biology. Your task is to identify all protein-protein "
                "interactions (PPI's) in the text, focusing on proteins involved in signaling pathways.",
                "Now review your extracted protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain only signalling pathway interactions "
                "and remove the rest.",
                "Review one more time the protein-protein interactions (PPI's) to  "
                "determine whether there are in the list regulations that are of a transcriptional or gene  "
                "regulatory nature. Retain those interactions that are only specific to PPI's in cell  "
                "signalling and remove those relations that represent relations betweentranscription factors "
                "to their gene targets.",
            ],
            "tf": [
                "You are a top-tier molecular biologist specialized in the field of "
                "cardiology and molecular biology. Your task is to identify all transcription factor (TF) "
                "to gene relations in the text. The source of the relations that you identify should be a "
                "TF, while the target should be genes who are regulated by the TF.",
                "Now review your extracted transcription factor (TF) to gene relations to determine if "
                "they are specific to gene regulatory networks. Retain those interactions that are only "
                "involving TF's and their gene targets and remove those that are not.",
                "Review one more time the transcription factor to gene relations  "
                "to determine whether there are in the list relations that are protein-protein "
                "interactions (PPI's) network or involved in protein signalling networks. Retain interactions  "
                "of gene regulatory networks involve a transcription factor and the gene whose expression  "
                "they regulate. Remove those relations that involve interactions between two signalling protein "
                "and PPI's.",
            ],
        },
        "complex": {
            "ppi": [
                "You are a top-tier molecular biologist specialized in the field of  "
                "cardiology and molecular biology. Your task is to identify all protein-protein  "
                "interactions (PPI's) in the text, focusing on proteins involved in signaling pathways.  "
                "Specify also the type of interaction (e.g., binding, activation, inhibition,  "
                "phosphorylation) and direction if applicable (e.g., protein A activates protein B).",
                "Now review the protein-protein interactions (PPI's) to determine if "
                "they are specific to signaling pathways. Retain those PPI's that are specific to "
                "signalling and remove those that are not.",
                "Review one more time the protein-protein interactions (PPI'S) to determine"
                "whether there are in the list regulations that are of a transcriptional or gene regulatory"
                "nature. Retain those interactions that are only specific to PPI's in cell signalling and remove "
                "those relations that show transcription factor to their gene targets.",
            ],
            "tf": [
                "You are a top-tier molecular biologist specialized in the field of "
                "cardiology and molecular biology. Your task is to identify all transcription factor (TF) to "
                "gene relations in the text. The source of the relations that you identify should be a "
                "TF, while the target should be genes who are regulated by the TF. Specify also the type of "
                "the relation (e.g. regulation, suppression, expression, etc.).",
                "Now review the transcription factor (TF) to gene relations to determine if  "
                "they are specific to gene regulatory networks. Retain those relations that are only "
                "involving TF's and their gene targets and remove those that are not.",
                "Review one more time the transcription factor (TF) to gene relations "
                "to determine whether there are in the list relations that are of a protein-protein "
                "interaction (PPI's) or protein signalling nature. Retain those relations which involve "
                "a transcription factor and the gene whose expression they regulate. Remove those "
                "relations that involve interactions between two signalling protein and PPI's.",
            ],
        },
    },
    2: {
        "simple": {
            "both": [
                "You are a top-tier molecular biologist specialized in the field of  "
                "cardiology and molecular biology. Now, in step 1, your task is to identify all protein-protein "
                "interactions (PPI's) involved in signalling as well as relations between transcription  "
                "factors (TF) and their target genes of a gene regulatory network.",
                "Now, in step 2,  review the extracted relations from step 1. Please retain those relations which "
                "correspond to protein-protein interactions (PPI's) that are involved in "
                "cell-signalling (e.g. through binding, activation, inhibition, phosphorylation, "
                "etc.). Please remove those transcription factor to gene relations that are involved "
                "in gene regulatory networks (e.g. regulation of expression or suppression of a gene).",
                "Review the extracted relations from step 1 again. Please retain those transcription factor (TF) "
                "to gene relations that are involved in gene regulatory networks (e.g. regulation of "
                "expression or suppression of a gene). Please remove those relations which correspond to "
                "protein-protein interactions that are involved in cell-signalling (e.g. through binding, "
                "activation, inhibition, phosphorylation, etc.).",
            ]
        },
        "complex": {
            "both": [
                "You are a top-tier molecular biologist specialized in the field of  "
                "cardiology and molecular biology. Now, in step 1, your task is to identify all protein-protein "
                "interactions (PPI's) involved in signalling as well as relations between transcription  "
                "factors (TF) and their target genes of a gene regulatory network.",
                "Now, in step 2, review the extracted relations. Please retain those relations which "
                "correspond to protein-protein interactions (PPI's) that are involved in "
                "cell-signalling (e.g. through binding, activation, inhibition, phosphorylation, "
                "etc.). Please remove those transcription factor (TF) to gene relations that are involved "
                "in gene regulatory networks (e.g. regulation of expression or suppression of a gene). "
                "For context, here is a good example to fulfil this request: "
                "'JAKs phosphorylate cytokine receptors which can bind a protein called Grb2. "
                "Grb2 then activates SOS proteins which stimulate MAPK signalling. MAPK can also "
                "phosphorylate STATs. Phosphorylated cytokine receptors can also be bound by PI3K, "
                "which allows activation of AKT. There is a broad range of genes that are regulated "
                "by STATs and none likely have as many broad functions as c-Fos and HIF-1α. STAT3 "
                "target genes include cyclin D1, BclXL, c-Myc, β-catenin, nuclear factor-κB (NF-κB)'."
                "Here a good output would be giving relations involving protein interactions such as: "
                "JAK phosphorylates cytokine receptors; cytokine receptors bind to Grb2; Grb2 activates "
                "SOS; SOS stimulates MAPK; MAPK phosphorylates STATS; Grb2 activates SOS; "
                "cytokine receptors binds PI3K; PI3K activates AKT."
                "Here a bad output would be giving relations involving gene regulatory relations such as:"
                "STATs regulates c-Fos; STATs regulate HIF-1α; STAT3 targets Cyclin D1; STAT3 targets BclXL; "
                "STAT3 targets y-Myc; STAT3 targets β-catenin; STAT3 targets NF-κB.",
                "Review the extracted relations from step 1 again. Please retain those transcription factor (TF) "
                "to gene relations that are involved in gene regulatory networks (e.g. regulation of  "
                "expression or suppression of a gene). Please remove those relations which correspond to  "
                "protein-protein interactions (PPI's) that are involved in cell-signalling (e.g. through  "
                "binding, activation, inhibition, phosphorylation, etc.). "
                "For context, here is a good example to fulfil this request: "
                "'JAKs phosphorylate cytokine receptors which can bind a protein called Grb2.  "
                "Grb2 then activates SOS proteins which stimulate MAPK signalling. MAPK can also  "
                "phosphorylate STATs. Phosphorylated cytokine receptors can also be bound by PI3K,  "
                "which allows activation of AKT. There is a broad range of genes that are regulated  "
                "by STATs and none likely have as many broad functions as c-Fos and HIF-1α. STAT3  "
                "target genes include cyclin D1, BclXL, c-Myc, β-catenin, nuclear factor-κB (NF-κB)'. "
                "Here a good output would be giving relations involving gene regulatory relations such as: "
                "STATs regulates c-Fos; STATs regulate HIF-1α; STAT3 targets Cyclin D1; STAT3 targets BclXL; "
                "STAT3 targets y-Myc; STAT3 targets β-catenin; STAT3 targets NF-κB. "
                "Here a bad output would be giving relations involving protein interactions such as: "
                "JAK phosphorylates cytokine receptors; cytokine receptors bind to Grb2; Grb2 activates "
                "SOS; SOS stimulates MAPK; MAPK phosphorylates STATS; Grb2 activates SOS; "
                "cytokine receptors binds PI3K; PI3K activates AKT.",
            ],
        },
    },
}
