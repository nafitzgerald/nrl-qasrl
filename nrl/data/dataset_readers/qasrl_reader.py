import codecs
import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json
import gzip

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField, MetadataField, SpanField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

from nrl.common.span import Span
from nrl.data.util import AnnotatedSpan, cleanse_sentence_text

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("qasrl")
class QaSrlReader(DatasetReader):
    def __init__(self, 
                 token_indexers: Dict[str, TokenIndexer] = None,
                 has_provinence = False,
                 bio_labels = False,
                 slot_labels = None,
                 min_answers = 3,
                 min_valid_answers = 3,
                 question_sources = None):
        super().__init__(False)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._has_provinence = has_provinence
        self._bio_labels = bio_labels
        self._invalid_thresh = 0
        self._max_spans = None
        self._min_answers = min_answers
        self._min_valid_answers = min_valid_answers
        self._slot_labels = slot_labels or ["wh", "aux", "subj", "verb", "obj", "prep", "obj2"]
        if bio_labels:
            self._total_args = 0.
            self._skipped_args = 0.

        self._question_sources = question_sources

        self._num_verbs = 0
        self._no_ann = 0
        self._not_enough_answers = 0
        self._not_enough_valid_answers = 0
        self._instances = 0
        self._qa_pairs = 0

    @overrides
    def _read(self, file_list: str):
        instances = []
        for file_path in file_list.split(","):
            if file_path.strip() == "":
                continue
            file_path = cached_path(file_path)

            logger.info("Reading QASRL instances from dataset file at: %s", file_path)
            data = []
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            elif file_path.endswith(".json"):
                with codecs.open(file_path, 'r', encoding='utf8') as open_file:
                    for line in f:
                        data.append(json.loads(line))
            
            for item in data:
                sent_id = item["sentenceId"]
                sentence_tokens = item["sentenceTokens"]

                annotations = []
                for _, verb_entry in item["verbEntries"].items():
                    verb_index = verb_entry["verbIndex"]

                    self._num_verbs += 1

                    annotations = []
                    for _, question_label in verb_entry["questionLabels"].items():
                        answers = len(question_label["answerJudgments"])
                        valid_answers = len([ans for ans in question_label["answerJudgments"] if ans["isValid"]])

                        if self._question_sources is not None:
                            if not any([source.startswith(prefix) for source in question_label["questionSources"] for prefix in self._question_sources]):
                                continue

                        if answers < self._min_answers:
                            self._not_enough_answers += 1
                            continue
                        if valid_answers < self._min_valid_answers:
                            self._not_enough_valid_answers += 1
                            continue

                        slots = []
                        for l in self._slot_labels:
                            slots.append(question_label["questionSlots"][l])

                        provinence = list(question_label["questionSources"])[0]

                        spans = []
                        for ans in question_label["answerJudgments"]:
                            if ans["isValid"]:
                                for s in ans["spans"]:
                                    spans.append(Span(s[0], s[1]-1))
                        
                        self._qa_pairs += 1
                        annotations.append(AnnotatedSpan(slots = slots, all_spans = spans, provinence=provinence))

                    if annotations:
                        self._instances += 1
                        yield self._make_instance_from_text(sentence_tokens, verb_index, annotations = annotations, sent_id = sent_id)
                    else:
                        self._no_ann += 1

        logger.info("Produced %d instances"%self._instances)
        logger.info("\t%d Verbs"%self._num_verbs)
        logger.info("\t%d QA pairs"%self._qa_pairs)
        logger.info("\t%d no annotation"%self._no_ann)
        logger.info("\t%d not enough answers"%self._not_enough_answers)
        logger.info("\t%d not enough valid answers"%self._not_enough_valid_answers)

    def _make_instance_from_text(self, sent_tokens, pred_index, annotations = None, sent_id = None):
        instance_dict = {}

        if isinstance(sent_tokens, str):
            sent_tokens = sent_tokens.split()
        sent_tokens = cleanse_sentence_text(sent_tokens)
        text_field = TextField([Token(t) for t in sent_tokens], self._token_indexers)
        instance_dict['text'] = text_field
        instance_dict['predicate_indicator'] = SequenceLabelField([1 if i == pred_index else 0 for i in range(len(sent_tokens))], text_field)

        if annotations is not None:
            for i, slot_name in enumerate(self._slot_labels):
                span_slot = ListField([LabelField(ann.slots[i], label_namespace="slot_%s_labels"%slot_name) for ann in annotations for span in ann.all_spans])
                instance_dict['span_slot_%s'%slot_name] = span_slot

            labeled_span_field = ListField([SpanField(span.start(), span.end(), text_field) for ann in annotations for span in ann.all_spans])
            instance_dict['labeled_spans'] = labeled_span_field

            if self._bio_labels:
                bio_labels = ["O"] * len(sent_tokens)

                bio_labels[pred_index] = "B-V"

                for span in self._resolve_spans(annotations, pred_index):
                    bio_labels[span.start()] = "B-ARG"
                    for i in range(span.start()+1, span.end()+1):
                        bio_labels[i] = "I-ARG"
                instance_dict["bio_label"] = SequenceLabelField(bio_labels, text_field, label_namespace="bio_labels")

            instance_dict['annotations'] = MetadataField({'annotations':annotations})

        metadata = {'pred_index' : pred_index, 'sent_text': " ".join(sent_tokens)}
        if sent_id is not None:
            metadata['sent_id'] = sent_id
        instance_dict['metadata'] = MetadataField(metadata)

        return Instance(instance_dict)

    def _resolve_spans(self, annotations, pred_index):
        result = []
        covered = set()
        for ann in annotations:
            self._total_args += 1

            nonoverlapping = [span for span in ann.all_spans if not set(range(span.start(), span.end()+1)) & covered]
            if not nonoverlapping:
                self._skipped_args += 1
                continue
            picked = self._pick_span(nonoverlapping, pred_index)
            covered = covered | set(range(picked.start(), picked.end()+1))
            result.append(picked)
        return result

    def _pick_span(self, spans, pred_index):
        counter = Counter(spans)
        max_count = max(counter.values())
        max_items = [span for span, cnt in counter.items() if cnt == max_count]

        if len(max_items) == 1:
            return max_items[0]

        return max(max_items, key = lambda x: x.size())

    @classmethod
    def from_params(cls, params: Params) -> 'QaSrlReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', Params({"tokens": {"type": "single_id", "lowercase_tokens": True}})))
        has_provinence = params.pop("has_provinence", False)
        bio_labels = params.pop("bio_labels", False)

        min_answers = params.pop("min_answers", 3)
        min_valid_answers = params.pop("min_valid_answers", 3)

        question_sources = params.pop("question_sources", None)

        params.assert_empty(cls.__name__)
        return QaSrlReader(token_indexers=token_indexers, has_provinence = has_provinence, bio_labels=bio_labels, min_answers=min_answers, min_valid_answers=min_valid_answers, question_sources=question_sources)
