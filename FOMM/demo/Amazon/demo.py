from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.transform import resize
import torch, torcheia
import argparse
import imageio
from scipy.spatial import ConvexHull


def make_parser():
    parser = argparse.ArgumentParser("pytorch eia demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./fomm_eia.mp4", type=str, help="generated video path"
    )

    parser.add_argument("-c", "--pt-file-fomm", type=str, help="fomm traced model path")
    parser.add_argument("-k", "--pt-file-kp", type=str, help="kp traced model path")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    return parser


def normalize_kp(kp_source, kp_source_jacobian, kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian,
                 adapt_movement_scale=False, use_relative_movement=False, use_relative_jacobian=False):
    movement_scale = 1
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source[0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0].data.cpu().numpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    
    if use_relative_movement:
        kp_value_diff = (kp_driving - kp_driving_initial)
        kp_value_diff *= movement_scale
        kp_new = kp_value_diff + kp_source
        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving_jacobian, torch.inverse(kp_driving_initial_jacobian))
            kp_new_jacobian = torch.matmul(jacobian_diff, kp_source_jacobian)
    
    return kp_new, kp_new_jacobian


def main():
    args = make_parser().parse_args()

    #kp_detector = torch.jit.load(args.pt_file_kp, map_location=torch.device('cpu'))
    fomm_model = torch.jit.load(args.pt_file_fomm, map_location=torch.device('cpu'))
    torch._C._jit_set_profiling_executor(False)
    #kp_detector_eia = torcheia.jit.attach_eia(kp_detector, 0)
    fomm_model_eia = torcheia.jit.attach_eia(fomm_model, 0)
    '''
    source = imageio.imread(args.source)
    reader = imageio.get_reader(args.driving)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source = resize(source, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    source = torch.tensor(source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    '''
    source = torch.randn(1, 3, 256, 256)
    kp_norm = torch.randn(1, 10, 2)
    kp_norm_jacobian = torch.randn(1, 10, 2, 2)
    kp_source = torch.randn(1, 10, 2)
    kp_source_jacobian = torch.randn(1, 10, 2, 2)
    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            #kp_source, kp_source_jacobian = kp_detector_eia.forward(source)
            #kp_driving_initial, kp_driving_initial_jacobian = kp_detector_eia.forward(driving[:, :, 0])

            #total_frames = driving.shape[2]
            total_frames = 169
            pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
            #predictions = []
            for frame_idx in range(total_frames):
                #driving_frame = driving[:, :, frame_idx]
                tic = time.time()
                #kp_driving, kp_driving_jacobian = kp_detector_eia.forward(source)
                #kp_norm, kp_norm_jacobian = normalize_kp(kp_source, kp_source_jacobian, kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian,
                 #                                        adapt_movement_scale=args.adapt_scale, use_relative_movement=args.relative, use_relative_jacobian=args.relative)
                out = fomm_model_eia.forward(source, kp_norm, kp_norm_jacobian, kp_source, kp_source_jacobian)
                toc = time.time()
                #im = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0]
                #predictions.append(im)
                pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
                pbar.update(1)

    #imageio.mimsave(args.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    main()
