from typing import Dict, List, Optional, Set, Tuple

import torch
import networkx as nx
import random

from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

from nrl.common.span import Span

class ThresholdMetric(Metric):
    def __init__(self,
            thresholds = [.5],
            remove_overlap = False,
            prf_outfile = None,
            match_heuristic = None):
        self._thresholds = thresholds
        self._remove_overlap = remove_overlap
        self._prf_outfile = prf_outfile
        self._match_heuristic = match_heuristic or (lambda x, y: x == y)

        self.reset()

    def reset(self):
        self._gold_spans = 0.
        self._covered = [0.] * len(self._thresholds)
        self._correct = [0.] * len(self._thresholds)
        self._predicted = [0.] * len(self._thresholds)

        self._wh_totals = {}
        self._wh_covered = [{} for _ in self._thresholds]

        self._prf_scores = []

    def __call__(self,
            predicted_spans: torch.Tensor,
            gold_annotations: torch.Tensor,
            ):
        gold = []
        pred = []
        has_span = []

        for b in range(len(predicted_spans)):
            gold_spans = gold_annotations[b]['annotations']
            self._gold_spans += len(gold_spans)

            #for gold in gold_spans:
            #    gold.all_spans = random.sample(gold.all_spans, 2)

            for g in gold_spans:
                wh = g.slots[0]
                self._wh_totals.setdefault(wh, 0)
                self._wh_totals[wh] += 1

                if wh in ['what', 'who']:
                    self._wh_totals.setdefault("core", 0)
                    self._wh_totals["core"] += 1
                else:
                    self._wh_totals.setdefault("aux", 0)
                    self._wh_totals["aux"] += 1

            for i, t in enumerate(self._thresholds):
                pred_spans = [s[0] for s in predicted_spans[b] if s[1] >= t]
                self._predicted[i] += len(pred_spans)

                picked_gold, picked_pred = self.get_matches(pred_spans, gold_spans)

                for g, tup in picked_gold.items():
                    f1, pred, gold = tup
                    if self._match_heuristic(pred, gold):
                        self._covered[i] += 1

                        wh = g.slots[0]
                        self._wh_covered[i].setdefault(wh, 0)
                        self._wh_covered[i][wh] += 1
                        if wh in ['what', 'who']:
                            self._wh_covered[i].setdefault("core", 0)
                            self._wh_covered[i]["core"] += 1
                        else:
                            self._wh_covered[i].setdefault("aux", 0)
                            self._wh_covered[i]["aux"] += 1

                for pred, tup in picked_pred.items():
                    f1, g, gold = tup
                    if self._match_heuristic(pred, gold):
                        self._correct[i] += 1

        if self._prf_outfile is not None:
            for b in range(len(predicted_spans)):
                gold_spans = gold_annotations[b]['annotations']
                pred_spans = predicted_spans[b]
                
                for p in pred_spans:
                    pred_span, prob = p 
                    matched_gold = [gold for gold in gold_spans if any([self._match_heuristic(g, pred_span) for g in gold.all_spans])]
                    self._prf_scores.append((prob, matched_gold))
                
    def get_matches(self, pred_spans, gold_spans):
        gold_spans = [g for g in gold_spans if g.text != 'V']

        G = nx.Graph()
        for s in gold_spans + pred_spans:
            G.add_node(s)

        max_golds = {}
        max_pred_scores = {}
        for gold_span_tuple in gold_spans:
            for span in pred_spans:
                max_f1, max_gold = max([(g.overlap_f1(span), g) for g in gold_span_tuple.all_spans], key = lambda x: x[0])
                max_golds[(gold_span_tuple, span)] = (max_f1, max_gold)
                if self._match_heuristic(span, max_gold):
                    #G.add_edge(gold_span_tuple, span, weight = max_f1)
                    G.add_edge(gold_span_tuple, span, weight = 1)
                    if span not in max_pred_scores or max_f1 > max_pred_scores[span][0]:
                        max_pred_scores[span] = (max_f1, gold_span_tuple, max_gold)

        matching = nx.max_weight_matching(G)

        matching = [(g, p) for g, p in matching.items() if g in gold_spans]

        picked_gold = {}
        picked_pred = {}
        for g, p in matching:
            f1, max_gold = max_golds[(g, p)]
            picked_gold[g] = (f1, p, max_gold)
            picked_pred[p] = (f1, g, max_gold)

        for p, match in max_pred_scores.items():
            if p not in picked_pred:
                picked_pred[p] = match

        return picked_gold, picked_pred

    def get_metric(self, reset=False):

        all_metrics = {}

        for k, covered, correct, total, wh_covered in zip(self._thresholds, self._covered, self._correct, self._predicted, self._wh_covered):
            p = correct / total if total > 0 else 0.
            r = covered / self._gold_spans if self._gold_spans > 0 else 0.
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.

            all_metrics["precision-at-%.1f"%k] = p
            all_metrics["recall-at-%.1f"%k] = r
            all_metrics["fscore-at-%.1f"%k] = f

            for wh, count in self._wh_totals.items():
                cov = wh_covered[wh] if wh in wh_covered else 0.
                all_metrics["wh[%s]-recall-at-%.1f"%(wh, k)] = cov / count

        if reset:
            if self._prf_outfile is not None:
                best_prf = (0., 0., 0.)
                with open(self._prf_outfile, 'w') as out:
                    self._prf_scores.sort(key = lambda x: x[0], reverse = True)

                    curr_covered = set()
                    curr_correct = 0.
                    curr_counted = 0.
                    prev_prob = None
                    auc = 0.0
                    last_r = 0.0

                    for prob, matched in self._prf_scores:
                        if prob != prev_prob:
                            if prev_prob is not None:
                                p = curr_correct / curr_counted if curr_counted > 0 else 0.
                                r = float(len(curr_covered)) / self._gold_spans
                                f = (2 * p * r) / (p + r) if p + r > 0 else 0
                                out.write("%.3f\t%.5f\t%.5f\t%.5f\n"%(prev_prob, r, p, f))
                                auc += (r - last_r) * p
                                if f > best_prf[2]:
                                    best_prf = (p, r, f, prev_prob)
                                last_r = r
                            prev_prob = prob
                        curr_counted += 1
                        if matched:
                            curr_correct += 1
                            curr_covered.update(matched)

                    p = curr_correct / curr_counted if curr_counted > 0 else 0.
                    r = float(len(curr_covered)) / self._gold_spans
                    f = (2 * p * r) / (p + r)
                    if f > best_prf[2]:
                        best_prf = (p, r, f, prev_prob)
                    out.write("%.3f\t%.5f\t%.5f\t%.5f\n"%(prev_prob, r, p, f))
                all_metrics['AUC'] = auc
                all_metrics['best_P'] = best_prf[0]
                all_metrics['best_R'] = best_prf[1]
                all_metrics['best_F'] = best_prf[2]
                all_metrics['best_prob'] = best_prf[3]

            self.reset()

        return all_metrics
