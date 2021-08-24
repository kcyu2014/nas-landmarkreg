import torch.functional as F


def gumbel_softmax(logits, temperature, gumbel_dist):
    # IPython.embed()
    y = logits + gumbel_dist.sample(logits.size()).squeeze().cuda()
    return F.softmax(y / temperature, dim=-1)
