import sys
from itertools import islice

import torch

from core.utils.registry import registry


class MultiPrinter:
    def __init__(self, text_location):
        self.text_location = text_location
        self.f = open(text_location, 'a', encoding="utf-8")

    def write(self, msg):
        sys.__stdout__.write(msg)
        self.f.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


@registry.register_loss("ClassificationAccuracy")
def ClassificationAccuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = sum(predicted == labels)
    total = labels.size(0)

    return correct / total


class _SkippedDataloader:
    def __init__(self, dataloader, skip):
        self.dataloader = dataloader
        self.skip = skip

    def __iter__(self):
        yield from islice(self.dataloader, self.skip, None)

    def __len__(self):
        return len(self.dataloader) - self.skip
