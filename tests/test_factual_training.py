import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers

from ownllm.training.books import encode_sft
from ownllm.training.factual import critical_token_weights, weighted_token_loss


def test_critical_weights_align_with_shifted_answer_and_mask_prompt_padding():
    tokenizer = Tokenizer(models.WordLevel({'<unk>':0,'<pad>':1,'<eos>':2,
                                           '<|im_end|>':3},unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    row = {'instruction':'An unrelated value 999 must not be supervised.',
           'output':'### 1. General Overview\nWater: calculated source record.\n'
                    '### 2. Engineering Profile\n'
                    '* **Typical Value / Range:** 18.015 (Conditions: calculated.)\n'
                    '* **Unit / Scale:** Da'}
    _, labels = encode_sft(row,tokenizer,160)
    weights = critical_token_weights(row,tokenizer,160,12)
    assert weights.shape == labels.shape
    assert not weights[labels==-100].any()
    assert (weights[labels!=-100]==12).sum() == 5  # Water; 18, dot, 015; Da.
    assert (weights[labels!=-100]==1).any()
    # Truncation preserves precisely the same answer prefix and prompt boundary.
    _, short_labels = encode_sft(row,tokenizer,20)
    short_weights = critical_token_weights(row,tokenizer,20,12)
    assert not short_weights[short_labels==-100].any()


def test_uniform_weighting_matches_original_loss_and_factual_weight_changes_gradient():
    logits = torch.tensor([[[1.,0.],[0.,1.],[.5,.5]]],requires_grad=True)
    labels = torch.tensor([[0,0,-100]])
    uniform = weighted_token_loss(logits,labels,torch.ones_like(labels,dtype=torch.float))
    assert torch.allclose(uniform,F.cross_entropy(logits.flatten(0,1),labels.flatten()))
    uniform.backward()
    before = logits.grad.clone()
    logits.grad.zero_()
    weighted_token_loss(logits,labels,torch.tensor([[1.,12.,100.]])).backward()
    assert logits.grad[0,1].abs().sum() > before[0,1].abs().sum()
    assert not logits.grad[0,2].any()
    with pytest.raises(ValueError,match='no supervised'):
        weighted_token_loss(logits,labels,torch.zeros_like(labels,dtype=torch.float))
    with pytest.raises(ValueError,match='finite'):
        weighted_token_loss(logits,labels,torch.tensor([[1.,float('nan'),1.]]))
