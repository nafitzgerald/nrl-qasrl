# nrl-qasrl

This repository contains the code for the paper: "[Large-Scale QA-SRL Parsing](https://arxiv.org/abs/1805.05377)" by [Nicholas FitzGerald](http://nfitz.net), [Julian Michael](http://julianmichael.org/), [Luheng He](https://homes.cs.washington.edu/~luheng/) and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz).

Data can be found at: <http://qasrl.org>

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

# Training Models

Training the QA-SRL parser consists of three stages. The Span Detection and Question Prediction models are trained seperately. Then, a script is run in order to combine these two models into one.

## Training Span Detector

To train the Span Detection models, run:

```
python -m allennlp.run train {$CONFIG_FILE} --include-package nrl -s {$SAVE_DIRECTORY}
```

Two config files are included with this repository. These must be modified in order to point the `*_data_path` fields to your data directory.

1. `configs/train_span.json` - which reproduces the Span-based model from our paper.
2. `configs/train_span_elmo.json` - which is the same model but includes ELMo word representations. In order to run this you must first download the word representations from [here](https://allennlp.org/elmo) and modify the `model/text_field_embedder/elmo/*_file` fields to point to this.

## Training Question Detector

Training the Question Detector uses the same command as above but with these config files:

1. `configs/train_question.json` - which reproduces the Sequence model from our paper.
2. `configs/train_question_elmo` - which, like above, includes the ELMo word representation.

## Combining models

To combine these two trained models into one model which can then be run for prediction, run the following script:

```
python scripts/combine_models.py --span {$PATH_TO_SPAN_MODEL_DIRECTORY} --ques {$PATH_TO_QUESTION_MODEL_DIRECTORY} --out {$OUTPUT_TAR_GZ_FILE}
```

