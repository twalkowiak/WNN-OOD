import torch


class CroW(torch.nn.Module):
    def __init__(self):
        super(CroW, self).__init__()

    @staticmethod
    def forward(x):
        spatial_a = 2.0
        spatial_b = 2.0

        fea = x
        if fea.ndimension() == 4:
            spatial_weight = fea.sum(dim=1, keepdims=True)
            z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdims=True)
            z = z ** (1.0 / spatial_a)
            spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

            c, w, h = fea.shape[1:]
            non_zeros = (fea != 0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
            channel_weight = torch.log(non_zeros.sum(dim=1, keepdims=True) / non_zeros)

            fea = fea * spatial_weight
            fea = fea.sum(dim=(2, 3))
            fea = fea * channel_weight

            return fea
        else:
            raise Exception("CroW: Wrong dimension")
