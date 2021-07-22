import torch
import torch.nn as nn
import numpy as np
from config import device


class JSD_Loss(nn.Module):
    def __init__(self):
        super(JSD_Loss, self).__init__()

    def entropy_multi(self, p, q):
        return torch.sum(p * torch.log(p / q), dim=0)

    def KLD(self, pk, qk):
        """
        Part to find KL divergence
        :param pk: torch.FloatTensor()
        :param qk: torch.FloatTensor()
        :return:
        """
        eps = torch.zeros(pk.shape) + 1e-12
        pk = pk + eps.to(device)
        qk = qk + eps.to(device)
        # normalize
        pk = 1.0 * pk / torch.sum(pk, dim=0)
        qk = 1.0 * qk / torch.sum(qk, dim=0)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same batch.")
        elif qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        return torch.sum(self.entropy_multi(pk, qk), dim=0)

    def forward(self, output, labels):
        """
        cross_entropy labels
        :param output: [B, 10000]
        :param labels: [B, 10000]
        :return: JSD(output, labels)
        """
        # pk = output.unsqueeze(-1)
        pk = output
        qk = labels.type(torch.float32)
        m = (pk + qk) / 2
        kl_pm = self.KLD(pk, m)
        kl_qm = self.KLD(qk, m)
        jsd = (kl_pm + kl_qm)/2
        ret = torch.sum(jsd)
        return ret


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(100,100)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Wasserstein Distance
class EMD_Loss(nn.Module):
    def __init__(self):
        super(EMD_Loss, self).__init__()
        self.discriminator = Discriminator()
    
    
    def forward(self, output, labels):
        # output.shape : [B, 100, 100]
        # label.shape : [B, 100, 100]
        pk = output
        qk = labels.type(torch.float32)
        loss = -torch.mean(self.discriminator(pk)) + torch.mean(self.discriminator(qk))

        return loss


