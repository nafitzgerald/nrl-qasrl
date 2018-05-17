from typing import Dict, List, Optional, Tuple

class AnnotatedSpan:
    def __init__(self, text = None, span = None, slots=None, all_spans=None, pred_index = None, provinence = None):
        self.text = text
        self.span = span
        self.slots = slots
        self.all_spans = all_spans
        self.pred_index = pred_index
        self.provinence = provinence

def cleanse_sentence_text(sent_text):
    sent_text = ["?" if w == "/?" else w for w in sent_text]
    sent_text = ["." if w == "/." else w for w in sent_text]
    sent_text = ["-" if w == "/-" else w for w in sent_text]
    sent_text = ["(" if w == "-LRB-" else w for w in sent_text]
    sent_text = [")" if w == "-RRB-" else w for w in sent_text]
    sent_text = ["[" if w == "-LSB-" else w for w in sent_text]
    sent_text = ["]" if w == "-RSB-" else w for w in sent_text]
    return sent_text

