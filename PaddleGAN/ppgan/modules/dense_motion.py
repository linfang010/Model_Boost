# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .grid_sample import bilinear_grid_sample
from .first_order import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


def adj(M):
    result = np.zeros((M.shape[0], M.shape[1])).astype('float32')
    for i in range(1, M.shape[0] + 1):
        for j in range(1, M.shape[1] + 1):
            result[j - 1, i - 1] = np.power(-1, i + j) #* M[1 - i - 1, 1 - j - 1]
    M = M.flatten().flip(0).reshape([2, 2]).t()
    result = paddle.to_tensor(result)
    result = result * M
    return result


def invmat(M):
    # return 1.0 / det(M) * adj(M)
    _det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return 1.0 / _det * adj(M)


class DenseMotionNetwork(nn.Layer):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """
    def __init__(self,
                 block_expansion,
                 num_blocks,
                 max_features,
                 num_kp,
                 num_channels,
                 estimate_occlusion_map=False,
                 scale_factor=1,
                 kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_kp + 1) *
                                   (num_channels + 1),
                                   max_features=max_features,
                                   num_blocks=num_blocks)

        self.mask = nn.Conv2D(self.hourglass.out_filters,
                            num_kp + 1,
                            kernel_size=(7, 7),
                            padding=(3, 3))

        self.occlusion = nn.Conv2D(self.hourglass.out_filters,
                                1,
                                kernel_size=(7, 7),
                                padding=(3, 3))

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        self.down = AntiAliasInterpolation2d(num_channels,
                                                self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving_value,
                                       kp_source_value):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving_value,
                                       spatial_size=spatial_size,
                                       kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source_value,
                                      spatial_size=spatial_size,
                                      kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = paddle.zeros(
            [heatmap.shape[0], 1, spatial_size[0], spatial_size[1]],
            heatmap.dtype)  #.type(heatmap.type())
        heatmap = paddle.concat([zeros, heatmap], axis=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source_value.dtype)
        identity_grid = identity_grid.reshape([1, 1, h, w, 2])
        coordinate_grid = identity_grid - kp_driving_value.reshape([bs, self.num_kp, 1, 1, 2])

        #jacobian = paddle.matmul(kp_source['jacobian'], paddle.inverse(kp_driving['jacobian']))
        jacobian = paddle.matmul(kp_source_jacobian, 
                                 paddle.stack([invmat(kp_driving_jacobian[0, i]) for i in range(10)], axis=0).unsqueeze(0))
        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
        # Todo: fix bug of paddle.tile
        p_jacobian = jacobian.reshape([bs, self.num_kp, 1, 1, 4])
        paddle_jacobian = paddle.tile(p_jacobian, [1, 1, h, w, 1])
        paddle_jacobian = paddle_jacobian.reshape([bs, self.num_kp, h, w, 2, 2])

        coordinate_grid = paddle.matmul(paddle_jacobian, coordinate_grid.unsqueeze(-1))
        coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source_value.reshape([bs, self.num_kp, 1, 1, 2])

        #adding background feature
        identity_grid = paddle.tile(identity_grid, (bs, 1, 1, 1, 1))
        sparse_motions = paddle.concat([identity_grid, driving_to_source], axis=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = paddle.tile(source_image.unsqueeze(1).unsqueeze(1),[1, self.num_kp + 1, 1, 1, 1, 1])  #.repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.reshape([bs * (self.num_kp + 1), -1, h, w])
        sparse_motions = sparse_motions.reshape((bs * (self.num_kp + 1), h, w, -1))
        
        # sparse_deformed = F.grid_sample(source_repeat, sparse_motions, mode='bilinear', padding_mode='zeros', align_corners=True)
        sparse_deformed = bilinear_grid_sample(source_repeat, sparse_motions, align_corners=True)
        sparse_deformed = sparse_deformed.reshape((bs, self.num_kp + 1, -1, h, w))
        
        return sparse_deformed

    def forward(self, source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian):
        source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving_value, kp_source_value)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving_value, kp_driving_jacobian, 
                                                   kp_source_value, kp_source_jacobian)
        
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)

        input = paddle.concat([heatmap_representation, deformed_source], axis=2)
        input = input.reshape([bs, -1, h, w])

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, axis=1)
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.transpose([0, 1, 4, 2, 3])
        deformation = (sparse_motion * mask).sum(axis=1)
        deformation = deformation.transpose([0, 2, 3, 1])

        # Sec. 3.2 in the paper
        occlusion_map = F.sigmoid(self.occlusion(prediction))

        return deformed_source, mask, deformation, occlusion_map

