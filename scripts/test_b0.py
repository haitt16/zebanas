from torchvision.models import efficientnet_b0
import torch

model = efficientnet_b0(num_classes=10)
input = torch.rand(1, 3, 224, 224)

from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, input)
print("FLOPS", flops.total()/1e6)

print("Params", sum(p.numel() for p in model.parameters() if p.requires_grad))

params = 0.
for n, p in model.named_parameters():
    if "features.7" in n:
        if p.requires_grad:
            params += p.numel()
            print(n, p.size())
print("Params cell", params)
