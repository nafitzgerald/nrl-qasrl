from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

from nrl.modules.question_generator.question_generator import QuestionGenerator
from nrl.metrics.question_prediction_metric import QuestionPredictionMetric

@Model.register("question_predictor")
class QuestionPredictor(Model):
    def __init__(self, vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                question_generator: QuestionGenerator,
                stacked_encoder: Seq2SeqEncoder = None,
                predicate_feature_dim: int = 100,
                dim_hidden: int = 100,
                embedding_dropout: float = 0.0,
                initializer: InitializerApplicator = InitializerApplicator(),
                regularizer: Optional[RegularizerApplicator] = None):
        super(QuestionPredictor, self).__init__(vocab, regularizer)

        self.dim_hidden = dim_hidden

        self.text_field_embedder = text_field_embedder
        self.predicate_feature_embedding = Embedding(2, predicate_feature_dim)

        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.stacked_encoder = stacked_encoder

        self.span_extractor = EndpointSpanExtractor(self.stacked_encoder.get_output_dim(), combination="x,y")

        self.question_generator = question_generator
        self.slot_labels = question_generator.get_slot_labels()

        self.question_metric = QuestionPredictionMetric(vocab, question_generator.get_slot_labels())

    @overrides
    def forward(self, 
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                labeled_spans: torch.LongTensor,
                **kwargs):
        span_mask = (labeled_spans[:, :, 0] >= 0).long()

        span_slot_labels = []
        for i, n in enumerate(self.slot_labels):
            if 'span_slot_%s'%n in kwargs and kwargs['span_slot_%s'%n] is not None:
                span_slot_labels.append(kwargs['span_slot_%s'%n] * span_mask)
        if len(span_slot_labels) == 0:
            span_slot_labels = None

        embedded_text_input = self.embedding_dropout(self.text_field_embedder(text))
        mask = get_text_field_mask(text)
        embedded_predicate_indicator = self.predicate_feature_embedding(predicate_indicator.long())
 
        embedded_text_with_predicate_indicator = torch.cat([embedded_text_input, embedded_predicate_indicator], -1)
        batch_size, sequence_length, embedding_dim_with_predicate_feature = embedded_text_with_predicate_indicator.size()

        if self.stacked_encoder.get_input_dim() != embedding_dim_with_predicate_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")

        encoded_text = self.stacked_encoder(embedded_text_with_predicate_indicator, mask)

        span_reps = self.span_extractor(encoded_text, labeled_spans, sequence_mask=mask, span_indices_mask = span_mask)

        output_dict = {}
        slot_logits = self.question_generator(span_reps, slot_labels=span_slot_labels)
        for i, n in enumerate(self.slot_labels):
            # Replace scores for padding and unk
            slot_logits[i][:,:,0:2] -= 9999999

            output_dict["slot_logits_%s"%n] = slot_logits[i]

        loss = None
        if span_slot_labels is not None:
            for i, n in enumerate(self.slot_labels):
                slot_loss = sequence_cross_entropy_with_logits(slot_logits[i], span_slot_labels[i], span_mask.float())
                if loss is None:
                    loss = slot_loss
                else:
                    loss += slot_loss
            self.question_metric(slot_logits, span_slot_labels, labeled_spans, mask=span_mask, sequence_mask=mask)
            output_dict["loss"] = loss

        output_dict['span_mask'] = span_mask

        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.question_metric.get_metric(reset=reset)
        if self.training:
            metric_dict = {x: y for x, y in metric_dict.items() if "word-accuracy" not in x or x == "word-accuracy-overall"}

        return metric_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        span_mask = output_dict['span_mask'].data.cpu()
        batch_size, num_spans = span_mask.size()

        slot_preds = []
        for l in self.slot_labels:
            maxinds = output_dict['slot_logits_%s'%(l)].data.cpu().max(-1)[1]
            slot_preds.append(maxinds)

        questions = []
        for b in range(batch_size):
            batch_questions = []
            for i in range(num_spans):
                if span_mask[b, i] == 1:

                    slots = []
                    for l, n in enumerate(self.slot_labels):
                        slot_word = self.vocab.get_index_to_token_vocabulary("slot_%s"%n)[int(slot_preds[l][b, i])]
                        slots.append(slot_word)

                    slots = tuple(slots)

                    batch_questions.append(slots)

            questions.append(batch_questions)
        
        output_dict['questions'] = questions
        return output_dict




    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BIOLabeler':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        predicate_feature_dim = params.pop("predicate_feature_dim", 100)
        dim_hidden = params.pop("hidden_dim", 100)

        question_generator = QuestionGenerator.from_params(vocab, params.pop("question_generator"))



        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   question_generator=question_generator,
                   predicate_feature_dim=predicate_feature_dim,
                   dim_hidden=dim_hidden,
                   initializer=initializer,
                   regularizer=regularizer)
