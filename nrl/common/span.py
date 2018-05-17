from typing import List

class Span:
    def __init__(self, start : int, end : int) -> None:
        assert start <= end, "The end index of a span must be >= the start index (got start=%d, end=%d)"%(start, end)
        self._start = start
        self._end = end

    def __str__(self):
        return "(%d, %d)"%(self._start, self._end)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def overlaps(self, other : 'Span'):
        return (self._end >= other._start and self._start <= other._start) or (other._end >= self._start and other._start <= self._end)

    def centroid(self) -> float:
        return float(self._start + self._end) / 2

    def start(self):
        return self._start

    def end(self):
        return self._end

    def size(self):
        return self._end - self._start + 1

    def overlap_precision(self, other):
        intersection = Span.intersection(self, other)
        int_size = intersection.size() if intersection is not None else 0
        tp = int_size
        fp = self.size() - int_size
        return float(tp) / float(tp + fp) if tp + fp > 0 else 0.

    def overlap_recall(self, other):
        intersection = Span.intersection(self, other)
        int_size = intersection.size() if intersection is not None else 0
        tp = int_size
        fn = other.size() - int_size
        assert tp >= 0, intersection
        assert fn >= 0, "%s\n%s\n%s\n%s"%(self, other, intersection, int_size)
        return float(tp) / float(tp + fn) if tp+fn > 0 else 0.

    def overlap_f1(self, other):
        p = self.overlap_precision(other)
        r = self.overlap_recall(other)
        f = (2 * p * r) / (p + r) if p + r > 0. else 0.
        return f

    def iou(self, other):
        intersect = Span.intersection(self, other)
        i = float(intersect.size()) if intersect is not None else 0.
        union = float(self.size()) + float(other.size()) - i
        iou = i / union
        return iou

    @classmethod
    def from_qasrl_string(cls, string:str) -> List['Span']:
        spans = [Span.from_hyphenated_string(s) for s in string.split(';')]
        spans.sort(key=lambda x: x.start())
        curr = spans[0]
        collapsed = []
        for i in range(1, len(spans)):
            next = spans[i]
            if curr.end()+1 == next.start():
                curr = Span(curr.start(), next.end())
            else:
                collapsed.append(curr)
                curr = next
        collapsed.append(curr)
        return collapsed


    @classmethod
    def from_hyphenated_string(cls, s:str) -> 'Span':
        start, end = s.split('-')
        return Span(int(start), int(end))

    @classmethod
    def union(cls, span1 : 'Span', span2 : 'Span'):
        assert span1.overlaps(span2), "Spans must overlap to take the union."
        start = min(span1._start, span2._start)
        end = max(span1._end, span2._end)
        return Span(start, end)

    @classmethod
    def intersection(cls, span1 : 'Span', span2 : 'Span'):
        start = max(span1._start, span2._start)
        end = min(span1._end, span2._end)
        if start <= end:
            return Span(start,end)
        else:
            return None
