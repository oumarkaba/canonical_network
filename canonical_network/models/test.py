import torch
import torchsort


x = torch.arange(2*2*3).view(2, 2, 3)
s = torch.FloatTensor([[1, 2], [2, 1]])

print(x)
print(s)

permutation = s.argsort(dim=1)
permutation = permutation.unsqueeze(-1).expand_as(x)
rank = torchsort.soft_rank(s) - 1
rank = rank.unsqueeze(-1).expand_as(x)
idx = rank.long()
frac = rank.frac()

left = x.gather(1, idx)
right = x.gather(1, (idx + 1).clamp(max=1))
result = (1 - frac) * left + frac * right

print(rank)
print(permutation)

print(x.gather(1, permutation))
print(result)

print(torchsort.soft_rank(torch.zeros(1, 1) + 1000))