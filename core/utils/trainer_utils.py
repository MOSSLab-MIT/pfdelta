import sys

import torch

from core.utils.registry import registry


class MultiPrinter:
    def __init__(self, text_location):
        self.text_location = text_location
        self.f = open(text_location, 'w', encoding="utf-8")

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
