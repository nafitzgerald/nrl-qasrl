from typing import Dict, List, Optional, Set, Tuple

import torch

from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

class QuestionPredictionMetric(Metric):
    def __init__(self,
            vocabulary: Vocabulary,
            slot_labels: List[str],
            count_span : bool = False,
            fine_grained: bool = False):
        self._vocabulary = vocabulary
        self._bio_vocabulary = vocabulary.get_index_to_token_vocabulary("bio_labels")
        self._slot_labels = slot_labels
        self._count_span = count_span
        self._fine_grained = fine_grained

        self.reset()

    def reset(self):
        self._total_words = 0.
        self._total_questions = 0.
        self._correct = [0.] * len(self._slot_labels)
        self._partial_correct = 0.
        self._completely_correct = 0.
        self._completely_correct_with_span = 0.
        self._wh_correct = {}
        self._wh_total = {}


    def __call__(self,
            slot_logits: List[torch.Tensor],
            slot_labels: List[torch.Tensor],
            spans: torch.Tensor,
            bio_logit: torch.Tensor = None,
            mask: Optional[torch.Tensor] = None,
            sequence_mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.ones_like(slot_labels)

        gold = []
        pred = []
        has_span = []

        mask, sequence_mask, spans = self.unwrap_to_tensors(mask, sequence_mask, spans)
        if bio_logit is not None:
            bio_logit = self.unwrap_to_tensors(bio_logit)
        sequence_lengths = get_lengths_from_binary_sequence_mask(sequence_mask)

        gold_spans = [[(spans[b, i, 0], spans[b, i, 1]) for i in range(spans.size(1)) if mask[b, i] == 1] for b in range(spans.size(0))]

        if bio_logit is not None:
            argmax_bio = bio_logit.max(-1)[1]
        for i, l in enumerate(self._slot_labels):
            g = []
            p = []
            logits, gold_labels = self.unwrap_to_tensors(slot_logits[i], slot_labels[i])
            argmax_predictions = logits.max(-1)[1]
            for b in range(mask.size(0)):
                if bio_logit is not None:
                    bio_predictions = argmax_bio[b, :]
                    length = sequence_lengths[b]
                    spans = self._extract_spans(bio_predictions[:length].tolist())
                    spans = [s for s, tag in spans if tag != 'V']
                else:
                    spans = []

                for n in range(mask.size(1)):
                    if mask[b][n] == 1:
                        p.append(self._vocabulary.get_index_to_token_vocabulary("slot_%s_labels"%(l))[int(argmax_predictions[b, n])])
                        g.append(self._vocabulary.get_index_to_token_vocabulary("slot_%s_labels"%(l))[int(gold_labels[b, n])])
                        if i == 0:
                            if gold_spans[b][n] in spans:
                                has_span.append(True)
                            else:
                                has_span.append(False)
            gold.append(g)
            pred.append(p)

        gold_questions = []
        pred_questions = []
        for i in range(len(gold[0])):
            g = []
            p = []
            for w in range(len(slot_labels)):
                g.append(gold[w][i])
                p.append(pred[w][i])
            gold_questions.append(g)
            pred_questions.append(p)

        for i in range(len(gold_questions)):
            correct = True
            self._total_questions += 1
            for w in range(len(slot_labels)):
                self._total_words += 1
                g = gold_questions[i][w]
                p = pred_questions[i][w]

                if g == p:
                    self._correct[w] += 1
                else:
                    correct = False

            g = gold_questions[i]
            p = pred_questions[i]
            if g[0] == p[0] and g[2] == p[2] and g[4] == p[4] and g[6] == p[6]:
                self._partial_correct += 1

            wh = g[0]
            self._wh_total.setdefault(wh, 0.)
            self._wh_total[wh] += 1
            wh_type = "core" if wh in {"what", "who"} else "aux"
            self._wh_total.setdefault(wh_type, 0.)
            self._wh_total[wh_type] += 1

            if g[0] == p[0]:
                self._wh_correct.setdefault(wh, 0.)
                self._wh_correct[wh] += 1
                self._wh_correct.setdefault(wh_type, 0.)
                self._wh_correct[wh_type] += 1
            if correct:
                self._completely_correct += 1

                if has_span[i]:
                    self._completely_correct_with_span += 1

    def get_metric(self, reset=False):

        all_metrics = {}
        all_metrics["word-accuracy-overall"] = sum(self._correct) / self._total_words
        all_metrics["question-accuracy"] = self._completely_correct / self._total_questions
        all_metrics["partial-question-accuracy"] = self._partial_correct / self._total_questions

        if self._count_span:
            all_metrics["question-and-span-accuracy"] = self._completely_correct_with_span / self._total_questions

        for i, l in enumerate(self._slot_labels):
            all_metrics["word-accuracy-%s"%(l)] = self._correct[i] / self._total_questions

        if self._fine_grained:
            for wh, total in self._wh_total.items():
                correct = self._wh_correct.setdefault(wh, 0.)
                all_metrics["wh[%s]-accuracy"%wh] = correct / total

        if reset:
            self.reset()
        return all_metrics

    def _extract_spans(self, tag_sequence: List[int]) -> Set[Tuple[Tuple[int, int], str]]:
        """
        Given an integer tag sequence corresponding to BIO tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
        as otherwise it is possible to get a perfect precision score whilst still predicting
        ill-formed spans in addition to the correct spans.

        Parameters
        ----------
        tag_sequence : List[int], required.
            The integer class labels for a sequence.

        Returns
        -------
        spans : Set[Tuple[Tuple[int, int], str]]
            The typed, extracted spans from the sequence, in the format ((span_start, span_end), label).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        spans = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, integer_tag in enumerate(tag_sequence):
            # Actual BIO tag.
            string_tag = self._bio_vocabulary[integer_tag]
            bio_tag = string_tag[0]
            conll_tag = string_tag[2:]
            if bio_tag == "O" or conll_tag =="V":
                # The span has ended.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "U":
                # The U tag is used to indicate a span of length 1,
                # so if there's an active tag we end it, and then
                # we add a "length 0" tag.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                spans.add(((index, index), conll_tag))
                active_conll_tag = None
            elif bio_tag == "B":
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) The span is an I tag for a different conll annotation.
                # We'll process the previous span if it exists, but also
                # include this span. This is important, because otherwise,
                # a model may get a perfect F1 score whilst still including
                # false positive ill-formed spans.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag:
            spans.add(((span_start, span_end), active_conll_tag))
        return spans


