from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.transform import resize
import multiprocessing
import onnxruntime
import argparse
import imageio


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./fomm_onnx.mp4", type=str, help="generated video path"
    )

    parser.add_argument("-c", "--onnx-file-fomm", type=str, help="fomm onnx model path")
    parser.add_argument("-k", "--onnx-file-kp", type=str, help="kp onnx model path")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    return parser


def normalize_kp(kp_source, kp_source_jacobian, kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian,
                 adapt_movement_scale=False, use_relative_movement=False, use_relative_jacobian=False):
    movement_scale = 1
    if adapt_movement_scale:
        source_area = np.array([cv2.contourArea(cv2.convexHull(pts)) for pts in kp_source], dtype=kp_source.dtype)
        driving_area = np.array([cv2.contourArea(cv2.convexHull(pts)) for pts in kp_driving_initial], dtype=kp_driving_initial.dtype)
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    
    if use_relative_movement:
        kp_value_diff = (kp_driving - kp_driving_initial)
        kp_value_diff *= movement_scale
        kp_new = kp_value_diff + kp_source
        if use_relative_jacobian:
            jacobian_diff = np.matmul(kp_driving_jacobian, np.linalg.inv(kp_driving_initial_jacobian))
            kp_new_jacobian = np.matmul(jacobian_diff, kp_source_jacobian)
    
    return kp_new, kp_new_jacobian


def main():
    args = make_parser().parse_args()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    #sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    kp_detector = onnxruntime.InferenceSession(args.onnx_file_kp, sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    fomm_model = onnxruntime.InferenceSession(args.onnx_file_fomm, sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

    source = imageio.imread(args.source)
    #cv2_source = source.astype('float32') / 255
    #source = cv2.cvtColor(cv2_source, cv2.COLOR_BGR2RGB)
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
    source = np.transpose(source[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    driving = np.transpose(np.array(driving_video)[np.newaxis].astype(np.float32), (0, 4, 1, 2, 3))

    ort_inputs = {kp_detector.get_inputs()[0].name: source}
    kp_source, kp_source_jacobian = kp_detector.run(None, ort_inputs)
    ort_inputs = {kp_detector.get_inputs()[0].name: driving[:, :, 0]}
    kp_driving_initial, kp_driving_initial_jacobian = kp_detector.run(None, ort_inputs)

    total_frames = driving.shape[2]
    pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
    predictions = []
    for frame_idx in range(driving.shape[2]):
        driving_frame = driving[:, :, frame_idx]
        tic = time.time()
        ort_inputs = {kp_detector.get_inputs()[0].name: driving_frame}
        kp_driving, kp_driving_jacobian = kp_detector.run(None, ort_inputs)
        kp_norm, kp_norm_jacobian = normalize_kp(kp_source, kp_source_jacobian, kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian,
                                                 adapt_movement_scale=args.adapt_scale, use_relative_movement=args.relative, use_relative_jacobian=args.relative)
        ort_inputs = {fomm_model.get_inputs()[0].name: source,
                      fomm_model.get_inputs()[1].name: kp_norm,
                      fomm_model.get_inputs()[2].name: kp_norm_jacobian,
                      fomm_model.get_inputs()[3].name: kp_source,
                      fomm_model.get_inputs()[4].name: kp_source_jacobian,
                      }
        out = fomm_model.run(None, ort_inputs)[0]
        toc = time.time()
        im = np.transpose(out, [0, 2, 3, 1])[0]
        #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        #joinedFrame = np.concatenate((cv2_source, im, frame_face_save), axis=1)
        predictions.append(im)

        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)
    pbar.close()
    imageio.mimsave(args.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    main()
