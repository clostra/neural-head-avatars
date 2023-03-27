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
        self.C = texture_maps[0].shape[1]

    def forward(self, uv_coords, uv_idcs, subject_id):
        """
        uv_coords of shape
        :param uv_coords: R x 2 normalized to -1, ... +1
        :param uv_idcs: R
        :param subject_id: R
        :return: C x R
        """

        assert len(uv_coords.shape) == 2
        assert len(uv_idcs.shape) == 1

        R, _ = uv_coords.shape
        ret = torch.zeros(R, self.C, device=uv_coords.device, dtype=uv_coords.dtype)
        uv_subj_coords = torch.cat((uv_coords, subject_id[:, None]), -1) # R x 3
        for i, map in enumerate(self.maps):
            mask = (uv_idcs == i) # R
            ret[mask] = F.grid_sample(
                map.permute(1, 0, 2, 3)[None],   # 1 x C x M x H x W
                uv_subj_coords[None, None, None, mask],   # 1 x 1 x 1 x R' x 3
                padding_mode="border",
                align_corners=True # 1 x C x 1 x 1 x R'
            )[0, :, 0, 0].permute(1, 0)

        return ret.permute(1, 0)

