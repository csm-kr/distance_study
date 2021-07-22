import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

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
        
        self.img_shape = (100, 100)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
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
        self.discriminator = Discriminator().to(device)
    
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        # [b, 100, 100]
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    
    def forward(self, output, labels):
        # output.shape : [B, 100, 100]
        # label.shape : [B, 100, 100]
        pk = output
        qk = labels.type(torch.float32)

        # Loss weight for gradient penalty
        lambda_gp = 10

        pk_validity = self.discriminator(pk)
        qk_validity = self.discriminator(qk)

        gradient_penalty = self.compute_gradient_penalty(self.discriminator, pk.data, qk.data)
        loss = -torch.mean(pk_validity) + torch.mean(qk_validity)+ lambda_gp * gradient_penalty

        return loss


