import torch.nn as nn
import torch

class sCrossEntropyLoss(nn.Module):
    def __init__(self, e=0.1, reduction='sum'):
        super(sCrossEntropyLoss, self).__init__()
        self.method = reduction
        self.e = e
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        log_x = self.logsoftmax(x)
        num_classes = x.shape[-1]
        y = torch.zeros_like(log_x).scatter_(1, target.unsqueeze(1), 1)
        y = y.to(target.device)

        y = (1 - self.e) * y + self.e / num_classes
        loss = -y * log_x
        if self.method.lower() == 'sum':
            return loss.sum()
        elif self.method.lower() == 'mean':
            return loss.mean(0).sum()

        return loss

class BCE_focal_loss(nn.Module):
    def __init__(self, gamma=2, reduction='sum'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets > 0.5).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0-inputs)**self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs)**self.gamma * torch.log(1.0 - inputs + 1e-14)

        r = pos_loss + neg_loss
        if self.reduction == 'mean':
            return torch.mean(torch.sum(r, 1))
        return r.sum()

def main():
    loss_sum = nn.CrossEntropyLoss(reduction='sum')
    loss2_sum = sCrossEntropyLoss(e=0, reduction='sum')
    loss3_sum = sCrossEntropyLoss(reduction='sum')

    loss_mean = nn.CrossEntropyLoss(reduction='mean')
    loss2_mean = sCrossEntropyLoss(e=0, reduction='mean')
    loss3_mean = sCrossEntropyLoss(reduction='mean')

    x = torch.tensor([[0.5, 0.9, 0.3], [0.1014, 1.8978, 0.1829]])
    target = torch.tensor([1, 1])

    print(loss_sum(x, target), loss2_sum(x, target), loss3_sum(x, target))
    print(loss_mean(x, target), loss2_mean(x, target), loss3_mean(x, target))

if __name__ == "__main__":
    main()
