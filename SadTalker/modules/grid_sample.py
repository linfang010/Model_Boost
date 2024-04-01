import torch
import torch.nn.functional as F


def grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2


def grid_sampler_compute_source_index(coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord

    
def bilinear_grid_sample(image, grid, align_corners=False):

    N, C, D, H, W = image.shape
    Gn, Gd, Gh, Gw, _ = grid.shape
    assert N == Gn
    
    image = image.transpose(0, 1)
    im_padded = F.pad(image, pad=[1, 1, 1, 1, 1, 1], mode='constant', value=0)
                    
    x = grid[:, :, :, :, 0]
    y = grid[:, :, :, :, 1]
    z = grid[:, :, :, :, 2]
    
    # Unnormalize with align_corners condition
    ix = grid_sampler_compute_source_index(x, W, align_corners)
    iy = grid_sampler_compute_source_index(y, H, align_corners)
    iz = grid_sampler_compute_source_index(z, D, align_corners)
    
    ix_0 = torch.floor(ix).long()
    iy_0 = torch.floor(iy).long()
    iz_0 = torch.floor(iz).long()
    ix_1 = ix_0 + 1
    iy_1 = iy_0 + 1
    iz_1 = iz_0 + 1
    
    index = torch.zeros_like(ix_0)
    for j in range(ix.size(0)):
        index[j] += j

    tnw = ((ix_1 - ix) * (iy_1 - iy) * (iz_1 - iz))
    tne = ((ix - ix_0) * (iy_1 - iy) * (iz_1 - iz))
    tsw = ((ix_1 - ix) * (iy - iy_0) * (iz_1 - iz))
    tse = ((ix - ix_0) * (iy - iy_0) * (iz_1 - iz))
    bnw = ((ix_1 - ix) * (iy_1 - iy) * (iz - iz_0))
    bne = ((ix - ix_0) * (iy_1 - iy) * (iz - iz_0))
    bsw = ((ix_1 - ix) * (iy - iy_0) * (iz - iz_0))
    bse = ((ix - ix_0) * (iy - iy_0) * (iz - iz_0))

    ix_0 = torch.clamp(ix_0+1, min=0, max=W+1)
    ix_1 = torch.clamp(ix_1+1, min=0, max=W+1)
    iy_0 = torch.clamp(iy_0+1, min=0, max=H+1)
    iy_1 = torch.clamp(iy_1+1, min=0, max=H+1)
    iz_0 = torch.clamp(iz_0+1, min=0, max=D+1)
    iz_1 = torch.clamp(iz_1+1, min=0, max=D+1)

    I_tnw = im_padded[:, index, iz_0, iy_0, ix_0]
    I_tne = im_padded[:, index, iz_0, iy_0, ix_1]
    I_tsw = im_padded[:, index, iz_0, iy_1, ix_0]
    I_tse = im_padded[:, index, iz_0, iy_1, ix_1]
    I_bnw = im_padded[:, index, iz_1, iy_0, ix_0]
    I_bne = im_padded[:, index, iz_1, iy_0, ix_1]
    I_bsw = im_padded[:, index, iz_1, iy_1, ix_0]
    I_bse = im_padded[:, index, iz_1, iy_1, ix_1]

    out = I_tnw*tnw + I_tne*tne + I_tsw*tsw + I_tse*tse + I_bnw*bnw + I_bne*bne + I_bsw*bsw + I_bse*bse
    
    return out.transpose(1, 0).contiguous()


'''
if __name__ == '__main__':
    
    inp = torch.rand((2,3,2,2,2))
    grid = torch.rand((2,2,2,2,3))
    grid = 2*grid - 1
    outp = F.grid_sample(inp, grid)
    outp1 = bilinear_grid_sample(inp, grid)
    print(outp == outp1)
'''
    
