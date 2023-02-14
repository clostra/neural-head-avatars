from nha.util.general import count_tensor_entries

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiSubjectMultiTexture(nn.Module):
    """
    stores several per-subject uv texture maps and handles sampling from them based on uv idx and subject_id
    """

    def __init__(self, *texture_maps):
        """
        text
        :param texture_maps: list of texture tensors of shape M x C x H_i x W_i
        """

        for t in texture_maps:
            assert len(t.shape) == 4

        super().__init__()
        self.maps = nn.ParameterList([nn.Parameter(t, requires_grad=True) for t in texture_maps])
        self.C = texture_maps[0].shape[0]

    def forward(self, uv_coords, uv_idcs, subject_id):
        """
        uv_coords of shape
        :param uv_coords: N x H x W x 2 normalized to -1, ... +1
        :param uv_idcs: N x H x W
        :param subject_id: N
        :return: N x C x H x W
        """

        assert len(uv_coords.shape) == 4
        assert len(uv_idcs.shape) == 3
        assert uv_coords.shape[:-1] == uv_idcs.shape

        N, H, W, _ = uv_coords.shape
        ret = torch.zeros(N, H, W, self.C, device=uv_coords.device, dtype=uv_coords.dtype)
        for i, map in enumerate(self.maps):
            mask = (uv_idcs == i)
            ret[mask] = F.grid_sample(map[subject_id], uv_coords[mask].view(1, 1, -1, 2),
                                      padding_mode="border", align_corners=True)[0, :, 0, :].permute(1, 0)

        return ret.permute(0, 3, 1, 2)

