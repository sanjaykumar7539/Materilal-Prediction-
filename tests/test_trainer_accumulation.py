import torch
from ownllm.training.engine import Trainer

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w=torch.nn.Parameter(torch.tensor(1.0))
    def forward(self,x,targets=None,caches=None):
        loss=((self.w*x-targets)**2).mean()
        return x,loss,[]

def test_actual_microbatch_accumulation_matches_combined_mean():
    m=Tiny(); opt=torch.optim.SGD(m.parameters(),lr=0.1)
    t=Trainer(m,opt,grad_accum=4,amp="none",max_grad=100)
    t.train_batch([(torch.tensor([1.]),torch.tensor([0.])),(torch.tensor([1.]),torch.tensor([0.]))])
    assert abs(float(m.w.detach())-0.8)<1e-5
