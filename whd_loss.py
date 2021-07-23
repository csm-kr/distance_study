import torch
import torch.nn as nn
from config import device
import math
from sklearn.utils.extmath import cartesian
import numpy as np


def generaliz_mean(tensor, dim, p=-1, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res = torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y, r=2):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    differences = torch.abs(differences)
    distances = torch.sum(differences**r, -1)**(1/r)
    return distances


class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height=100,
                 resized_width=100,
                 p=-1):

        super().__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())
        self.p = p

    def map2coord(self, map, thres=1.0):
        '''
        (gaussian) map 을 받아서 threshold 보다 큰 값들을 1 로 바꾸어 주는 함수
        :param map:
        :return:
        '''
        # gt_map : [B, anchors]
        batch_size = map.size(0)
        mask_100_ = map.reshape(batch_size, -1)  # [B, 10000]
        mask_100 = (mask_100_ >= thres).type(torch.float32)  # [0, 1] 로 바꿔버리기

        nozero_100 = []
        batch_matrices_100 = []

        for b in range(batch_size):
            nozero_100.append(mask_100[b].nonzero().squeeze())
            coordinate_matrix_100 = torch.from_numpy(cartesian([np.arange(100), np.arange(100)]))  # [B, 100, 100, 2]
            batch_matrices_100.append(coordinate_matrix_100)

        coordinate_matries_100 = torch.stack(batch_matrices_100, dim=0)

        mask_100_vis = mask_100_.view(-1, 100, 100)
        mask_100_vis = mask_100.view(-1, 100, 100)

        # make seq gt
        seq_100 = []
        for b in range(batch_size):
               seq_100.append(coordinate_matries_100[b][nozero_100[b]].to(device))
        # print(len(seq_100[0]))
        return seq_100, mask_100_vis

    def forward(self, prob_map, gt_map, orig_sizes=torch.LongTensor([[100, 100], [100, 100]]).to(device)):

        gt, mask_100_vis = self.map2coord(map=gt_map)

        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        # original size 와 input size 가 같으므로
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            # gt_b = gt_b.type(torch.float32)
            # gt_b += torch.full_like(gt_b, 0.45)
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            # p = torch.randn([10000]).to(device)
            # p = torch.where(p>=0, p, torch.zeros_like(p))
            # p = torch.randint(0, 100000, size=[10000]).type(torch.float32)/50000
            # p *= 10
            # p = p.to(device)
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            term_1_ = (1 / (n_est_pts + 1e-6)) * torch.sum(torch.min(p_replicated * d_matrix, 1)[0])
            # print(torch.equal(p * torch.min(d_matrix, 1)[0], torch.min(p_replicated * d_matrix, 1)[0]))
            weighted_d_matrix = (1 - p_replicated) * self.max_dist # + p_replicated * d_matrix

            # minn = generaliz_mean(weighted_d_matrix,
            #                       p=self.p,
            #                       dim=0, keepdim=False)
            # term_2 = torch.mean(minn)

            # our method
            term_2 = torch.mean(torch.min(weighted_d_matrix, 0)[0])  # FIXME

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        res = terms_1.mean() + terms_2.mean()

        return res


class W_HausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height=100,
                 resized_width=100,
                 p=-1):

        super().__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())
        self.p = p

    def map2coord(self, map, thres=1.0):
        '''
        (gaussian) map 을 받아서 threshold 보다 큰 값들을 1 로 바꾸어 주는 함수
        :param map:
        :return:
        '''
        # gt_map : [B, anchors]
        batch_size = map.size(0)
        mask_100_ = map.reshape(batch_size, -1)  # [B, 10000]
        mask_100 = (mask_100_ >= thres).type(torch.float32)  # [0, 1] 로 바꿔버리기

        nozero_100 = []
        batch_matrices_100 = []

        for b in range(batch_size):
            nozero_100.append(mask_100[b].nonzero().squeeze())
            coordinate_matrix_100 = torch.from_numpy(cartesian([np.arange(100), np.arange(100)]))  # [B, 100, 100, 2]
            batch_matrices_100.append(coordinate_matrix_100)

        coordinate_matries_100 = torch.stack(batch_matrices_100, dim=0)

        mask_100_vis = mask_100_.view(-1, 100, 100)
        mask_100_vis = mask_100.view(-1, 100, 100)

        # make seq gt
        seq_100 = []
        for b in range(batch_size):
               seq_100.append(coordinate_matries_100[b][nozero_100[b]].to(device))
        print(len(seq_100[0]))
        return seq_100, mask_100_vis

    def forward(self, prob_map, gt_map, orig_sizes=torch.LongTensor([[100, 100], [100, 100]]).to(device)):

        gt, mask_100_vis = self.map2coord(map=gt_map)

        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        # original size 와 input size 가 같으므로
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * d_matrix

            minn = generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # our method
            # term_2 = torch.mean(torch.min(weighted_d_matrix, 0)[0])  # FIXME

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        res = terms_1.mean() + terms_2.mean()
        return res


class HausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height=100,
                 resized_width=100,
                 p=-1):

        super().__init__()

        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))

        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())
        self.p = p

    def map2coord(self, map, thres=0.5):
        '''
        (gaussian) map 을 받아서 threshold 보다 큰 값들을 1 로 바꾸어 주는 함수
        :param map:
        :return:
        '''
        # gt_map : [B, anchors]
        batch_size = map.size(0)


        mask_100_ = map.reshape(batch_size, -1)  # [B, 10000]
        mask_100 = (mask_100_ >= thres).type(torch.float32)  # [0, 1] 로 바꿔버리기



        nozero_100 = []
        batch_matrices_100 = []

        for b in range(batch_size):
            nozero_100.append(mask_100[b].nonzero().squeeze())
            coordinate_matrix_100 = torch.from_numpy(cartesian([np.arange(100), np.arange(100)]))  # [B, 100, 100, 2]
            batch_matrices_100.append(coordinate_matrix_100)

        coordinate_matries_100 = torch.stack(batch_matrices_100, dim=0)

        mask_100_vis = mask_100_.view(-1, 100, 100)
        mask_100_vis = mask_100.view(-1, 100, 100)

        # make seq gt
        seq_100 = []
        for b in range(batch_size):
               seq_100.append(coordinate_matries_100[b][nozero_100[b]].to(device))
        print(len(seq_100[0]))
        return seq_100, mask_100_vis

    def forward(self, prob_map, gt_map, orig_sizes=torch.LongTensor([[100, 100], [100, 100]]).to(device)):

        # indices = (prob_map[0] > 0.5).nonzero(as_tuple=False)
        mask_map = prob_map[0].masked_fill_((prob_map[0] < 0.5), 0)
        mask_map = mask_map.masked_fill_((prob_map[0] >= 0.5), 1)
        # .nonzero(as_tuple=False)
        # c = torch.count_nonzero(index).float()
        #
        # pred = [index]
        # pred, mask_100_vis_pred = self.map2coord(map=prob_map)
        gt, mask_100_vis = self.map2coord(map=gt_map)
        py_ = mask_100_vis
        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        # original size 와 input size 가 같으므로
        terms_1 = []
        terms_2 = []
        loss_1 = []
        w = self.all_img_locations


        prob_map_b = prob_map[0, :, :]
        px = prob_map_b.view(prob_map_b.nelement(), 1)
        px = px.repeat(1, 10000)
        py = py_.view(1, 10000).repeat(10000, 1)
        prob = (1 - px) * (1 - py)

        for b in range(batch_size):
            #prob_map_b = prob_map[b, :, :]
            #vpx = prob_map_b.view(prob_map_b.nelement())
            # pred_b = pred[b]
            gt_b = gt[b]
            loss = torch.abs((cdist(w, w) * px).mean(dim=1) - (cdist(w, w) * py).mean(dim=1))
            # orig_size_b = orig_sizes[b, :]
            # norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)
            #
            # # Corner case: no GT points
            # if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
            #     terms_1.append(torch.tensor([0],
            #                                 dtype=torch.get_default_dtype()))
            #     terms_2.append(torch.tensor([self.max_dist],
            #                                 dtype=torch.get_default_dtype()))
            #     continue
            #
            # # Pairwise distances between all possible locations and the GTed locations
            # n_gt_pts = gt_b.size()[0]
            # normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            # normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            # d_matrix = cdist(normalized_x, normalized_y)
            #
            # # Reshape probability map as a long column vector,
            # # and prepare it for multiplication
            # p = prob_map_b.view(prob_map_b.nelement())
            # n_est_pts = p.sum()
            # p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
            #
            # # Weighted Hausdorff Distance
            # term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            # weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * d_matrix
            #
            # minn = generaliz_mean(weighted_d_matrix,
            #                       p=self.p,
            #                       dim=0, keepdim=False)
            # term_2 = torch.mean(minn)

            # our method
            # term_2 = torch.mean(torch.min(weighted_d_matrix, 0)[0])  # FIXME

            # terms_1.append(term_1)
            # terms_2.append(term_2)
            loss_1.append(loss)

        # terms_1 = torch.stack(terms_1)
        # terms_2 = torch.stack(terms_2)
        # res = terms_1.mean() + terms_2.mean()
        loss_1 = torch.stack(loss_1)
        res = loss_1.mean()
        return res


class GeneralizedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height=100,
                 resized_width=100,
                 r=2):

        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())
        self.r = r

    def forward(self, prob_map, prob_y, gt, orig_sizes=torch.LongTensor([[100, 100], [100, 100]]).to(device)):

        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        # original size 와 input size 가 같으므로
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_y_ = prob_y[b, ...]
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y, self.r)

            n_gt_sum = prob_y_[(prob_y_ > 0.2)].sum()

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_x = p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
            p_y = (prob_y_[(prob_y_ > 0.2)]).unsqueeze(0).expand(d_matrix.size())

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(torch.min(p_x * p_y * d_matrix, 1)[0])

            dense_d_matrix = (1 - p_x) * self.max_dist + p_x * p_y * d_matrix
            term_2 = (1 / (n_gt_sum + 1e-6)) * torch.sum(torch.min(dense_d_matrix, 0)[0])  # FIXME 1

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        res = terms_1.mean() + terms_2.mean()
        return res


if __name__ == '__main__':
    prob_map = torch.randn([2, 10000])
    gt_depth_ = torch.randn([2, 120087])
    prob_map = prob_map.view(-1, 100, 100)
    whd_loss = WeightedHausdorffDistance(resized_width=100, resized_height=100)
    seq_0, seq_1 = whd_loss.map2coord(gt_depth_)
    loss = whd_loss.forward(prob_map, (seq_0, seq_1))
    print(loss)