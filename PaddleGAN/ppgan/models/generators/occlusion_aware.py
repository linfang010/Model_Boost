# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import paddle
from paddle import nn
import paddle.nn.functional as F
from ...modules.first_order import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, make_coordinate_grid
from ...modules.first_order import MobileResBlock2d, MobileUpBlock2d, MobileDownBlock2d
from ...modules.dense_motion import DenseMotionNetwork
from ...modules.grid_sample import bilinear_grid_sample
import numpy as np
import cv2


class OcclusionAwareGenerator(nn.Layer):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """
    def __init__(self,
                 num_channels,
                 num_kp,
                 block_expansion,
                 max_features,
                 num_down_blocks,
                 num_bottleneck_blocks,
                 estimate_occlusion_map=False,
                 dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        self.dense_motion_network = DenseMotionNetwork(
            num_kp=num_kp,
            num_channels=num_channels,
            estimate_occlusion_map=estimate_occlusion_map,
            **dense_motion_params)

        self.first = SameBlock2d(num_channels,
                                block_expansion,
                                kernel_size=(7, 7),
                                padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2**i))
            out_features = min(max_features, block_expansion * (2**(i + 1)))
            down_blocks.append(
                DownBlock2d(in_features,
                            out_features,
                            kernel_size=(3, 3),
                            padding=(1, 1)))
        self.down_blocks = nn.LayerList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features,
                                block_expansion * (2**(num_down_blocks - i)))
            out_features = min(
                max_features,
                block_expansion * (2**(num_down_blocks - i - 1)))
            up_blocks.append(
                UpBlock2d(in_features,
                            out_features,
                            kernel_size=(3, 3),
                            padding=(1, 1)))
        self.up_blocks = nn.LayerList(up_blocks)

        self.bottleneck = paddle.nn.Sequential()
        in_features = min(max_features, block_expansion * (2**num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_sublayer(
                'r' + str(i),
                ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
      
        self.final = nn.Conv2D(block_expansion,
                            num_channels,
                            kernel_size=(7, 7),
                            padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.pad = 5


    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape

        identity_grid = make_coordinate_grid((h, w), type=inp.dtype)
        identity_grid = identity_grid.reshape([1, h, w, 2])
        visualization_matrix = np.zeros((h, w)).astype("float32")
        visualization_matrix[self.pad:h - self.pad,
                                self.pad:w - self.pad] = 1.0
        gauss_kernel = paddle.to_tensor(
            cv2.GaussianBlur(visualization_matrix, (9, 9),
                                0.0,
                                borderType=cv2.BORDER_ISOLATED))
        gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(-1)
        deformation = gauss_kernel * deformation + (
            1 - gauss_kernel) * identity_grid

        return bilinear_grid_sample(inp, deformation, align_corners=True)

    def forward(self, source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        # output_dict = {}
        deformed_source, mask, deformation, occlusion_map = self.dense_motion_network(source_image, kp_driving_value,
                                                                                      kp_driving_jacobian,
                                                                                      kp_source_value,
                                                                                      kp_source_jacobian)
        # output_dict['mask'] = dense_motion['mask']
        # output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        # if 'occlusion_map' in dense_motion:
        #    occlusion_map = dense_motion['occlusion_map']
        #    output_dict['occlusion_map'] = occlusion_map
        # else:
        #    occlusion_map = None
        # deformation = dense_motion['deformation']
        out = self.deform_input(out, deformation)

        h, w = occlusion_map.shape[2:]
        occlusion_map[:, :, 0:self.pad, :] = 1.0
        occlusion_map[:, :, :, 0:self.pad] = 1.0
        occlusion_map[:, :, h - self.pad:h, :] = 1.0
        occlusion_map[:, :, :, w - self.pad:w] = 1.0
        out = out * occlusion_map

        # output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)
        # output_dict["prediction"] = out
        #
        # return output_dict
        return out
