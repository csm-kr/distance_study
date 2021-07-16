import torch
import torch.nn as nn


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
        # normalise
        pk = 1.0 * pk / torch.sum(pk, dim=0)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0 * qk / torch.sum(qk, dim=0)
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