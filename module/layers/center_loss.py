import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.
    Ref:
        A Discriminative Feature Learning Approach for Deep Face Recognition
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, device, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim)).to(device)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()

        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-10, max=1e+10)  # for numerical stability
            dist.append(value)

        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

