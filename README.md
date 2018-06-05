# nrl-qasrl

This repository contains the code for the paper: "Large-Scale QA-SRL Parsing" by Nicholas FitzGerald, Julian Michael, Luheng He and Luke Zettlemoyer.

Data can be found at: qasrl.org

More documentation to come...

# Prerequisites

This library requires [AllenNLP](https://github.com/allenai/allennlp).

# Pretrained model

A pretrained model can be downloaded from here: [Download Pretrained Model](https://drive.google.com/open?id=1FvMpjTfumVaSfwTOdWbJfEYFgGSAs0CS).

This model uses 8 LSTM-layers for both span detection and question generation, and a 4-layer LSTM generator.
It also uses ELMo deep contextualized word representations.

To run predictions with this model on new text, prepare a JSON-lines document with one line for each sentence with the following format:

```
{"sentence": "John went to the store."}
{"sentence": "The man ate the burrito and threw the trash in the garbage"}
```

Run prediction with the following command:

```
python -m allennlp.run predict {$PATH_TO_DOWNLOADED_MODEL}/qasrl_parser_elmo.tar.gz {$INPUT_FILE} --include-package nrl --predictor qasrl_parser --output-file {$OUTPUT_FILE}
```

Which will produce the following output:

```
{"words": ["John", "went", "to", "the", "store"], "verbs": [{"verb": "went", "qa_pairs": [{"question": "Who went somewhere?", "spans": ["John"]}, {"question": "Where did someone go?", "spans": ["to the store"]}], "index": 1}]}
{"words": ["The", "man", "ate", "the", "burrito", "and", "threw", "the", "trash", "in", "the", "garbage", "bin", "."], "verbs": [{"verb": "ate", "qa_pairs": [{"question": "Who ate something?", "spans": ["The man"]}, {"question": "What did someone eat?", "spans": ["the burrito"]}], "index": 2}, {"verb": "threw", "qa_pairs": [{"question": "Who threw something?", "spans": ["The man"]}, {"question": "What did someone throw?", "spans": ["the trash"]}, {"question": "Where did someone throw something?", "spans": ["in the garbage bin", "the garbage bin"]}], "index": 6}]}
```
