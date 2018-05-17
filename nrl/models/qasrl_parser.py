from typing import Dict

from overrides import overrides

import torch

from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.data import Vocabulary

from nrl.models.question_predictor import QuestionPredictor
from nrl.models.span_detector import SpanDetector

@Model.register("qasrl_parser")
class QaSrlParser(Model):
    def __init__(self, vocab: Vocabulary,
                 span_detector: SpanDetector,
                 question_predictor: QuestionPredictor):
        super(QaSrlParser, self).__init__(vocab)
        self.span_detector = span_detector
        self.question_predictor = question_predictor

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                labeled_spans: torch.LongTensor = None,
                annotations: Dict = None,
                **kwargs):
        raise NotImplementedException()

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QaSrlParser':
        span_detector = Model.from_params(vocab, params.pop("span_detector"))
        question_predictor = Model.from_params(vocab, params.pop("question_predictor"))

        return QaSrlParser(vocab, span_detector = span_detector, question_predictor = question_predictor)
