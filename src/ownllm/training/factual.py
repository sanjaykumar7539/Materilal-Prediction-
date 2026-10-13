"""Emphasize factual answer spans without assigning loss to prompt or padding."""
import re

import torch
import torch.nn.functional as F

from .books import instruction_prompt


def critical_token_weights(row, tokenizer, seq_len, factor):
    if factor < 1:
        raise ValueError('Critical field weight must be at least one')
    output = row['output']
    spans = []
    for pattern in (r'General Overview\n([^\n]+?): ',
                    r'Typical Value / Range:\*\* ([^\n]+?)(?= \(Conditions:|\n|$)',
                    r'Unit / Scale:\*\* ([^\n]+)'):
        spans.extend(match.span(1) for match in re.finditer(pattern, output))
    answer = tokenizer.encode(output)
    answer_weights = [factor if any(start < b and end > a for start,end in spans) else 1.
                      for a,b in answer.offsets] + [1.,1.]
    answer_weights = answer_weights[:seq_len]
    prefix_length = min(len(tokenizer.encode(instruction_prompt(row)).ids),
                        max(1, seq_len + 1 - len(answer_weights)))
    weights = ([0.] * prefix_length + answer_weights)[1:]
    return torch.tensor(weights + [0.] * (seq_len - len(weights)))


def weighted_token_loss(logits, labels, weights):
    if weights.shape != labels.shape or logits.shape[:2] != labels.shape:
        raise ValueError('Token weights must match target positions')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Token weights must be finite and nonnegative')
    active = weights * (labels != -100)
    denominator = active.sum()
    if denominator <= 0:
        raise ValueError('Weighted batch has no supervised tokens')
    losses = F.cross_entropy(logits.flatten(0,1), labels.flatten(),
                             ignore_index=-100, reduction='none').reshape_as(labels)
    return (losses * active).sum() / denominator
