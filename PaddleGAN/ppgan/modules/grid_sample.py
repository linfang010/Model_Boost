import paddle
import paddle.nn.functional as F


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def bilinear_grid_sample(im, grid, align_corners=True):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (paddle.Tensor): Input feature map, shape (N, C, H, W)
        grid (paddle.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        paddle.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """

    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape([n, -1])
    y = y.reshape([n, -1])

    x0 = paddle.floor(x)
    y0 = paddle.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)

    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = paddle.where(x0 < 0, paddle.to_tensor(0, dtype=x0.dtype), x0)
    x0 = paddle.where(x0 > padded_w - 1, paddle.to_tensor(padded_w - 1, dtype=x0.dtype), x0)
    x1 = paddle.where(x1 < 0, paddle.to_tensor(0, dtype=x1.dtype), x1)
    x1 = paddle.where(x1 > padded_w - 1, paddle.to_tensor(padded_w - 1, dtype=x1.dtype), x1)
    y0 = paddle.where(y0 < 0, paddle.to_tensor(0, dtype=y0.dtype), y0)
    y0 = paddle.where(y0 > padded_h - 1, paddle.to_tensor(padded_h - 1, dtype=y0.dtype), y0)
    y1 = paddle.where(y1 < 0, paddle.to_tensor(0, dtype=y1.dtype), y1)
    y1 = paddle.where(y1 > padded_h - 1, paddle.to_tensor(padded_h - 1, dtype=y1.dtype), y1)

    im_padded = im_padded.reshape([n, c, -1])

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand([-1, c, -1])
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand([-1, c, -1])
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand([-1, c, -1])
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand([-1, c, -1])

    Ia = paddle_gather(im_padded, 2, x0_y0)
    Ib = paddle_gather(im_padded, 2, x0_y1)
    Ic = paddle_gather(im_padded, 2, x1_y0)
    Id = paddle_gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape([n, c, gh, gw])
