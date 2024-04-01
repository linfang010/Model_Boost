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
#imageio.plugins.ffmpeg.download()


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./tpsmm_onnx.mp4", type=str, help="generated video path"
    )

    parser.add_argument("-c", "--onnx-file-tpsmm", type=str, help="tpsmm onnx model path")
    parser.add_argument("-k", "--onnx-file-kp", type=str, help="kp onnx model path")


    return parser


def calc_relative_kp(kp_source, kp_driver, kp_driver_ref, power = 1.0):
    source_area  = np.array([ cv2.contourArea(cv2.convexHull(pts)) for pts in kp_source ], dtype=kp_source.dtype)
    driving_area = np.array([ cv2.contourArea(cv2.convexHull(pts)) for pts in kp_driver_ref ], dtype=kp_driver_ref.dtype)
    movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    return kp_source + (kp_driver - kp_driver_ref) * movement_scale[:,None,None] * power


def create_transformations_params(kp_source, kp_driver):
    kp_num=10
    kp_sub_num=5

    kp_d = kp_driver.reshape(-1, kp_num, kp_sub_num, 2)
    kp_s = kp_source.reshape(-1, kp_num, kp_sub_num, 2)


    K = np.linalg.norm(kp_d[:,:,:,None]-kp_d[:,:,None,:], ord=2, axis=4) ** 2
    K = K * np.log(K+1e-9)

    kp_1d = np.concatenate([kp_d, np.ones(kp_d.shape[:-1], dtype=kp_d.dtype)[...,None] ], -1)

    P = np.concatenate([kp_1d, np.zeros(kp_d.shape[:2] + (3, 3), dtype=kp_d.dtype)], 2)
    L = np.concatenate([K,kp_1d.transpose(0,1,3,2)],2)
    L = np.concatenate([L,P],3)

    Y = np.concatenate([kp_s, np.zeros(kp_d.shape[:2] + (3, 2), dtype=kp_d.dtype)], 2)

    one = np.broadcast_to( np.eye(Y.shape[2], dtype=kp_d.dtype), L.shape)*0.01

    L = L + one

    param = np.matmul(np.linalg.inv(L),Y)

    theta = param[:,:,kp_sub_num:,:].transpose(0,1,3,2)
    control_points = kp_d
    control_params = param[:,:,:kp_sub_num,:]
    return theta, control_points, control_params


def main():
    args = make_parser().parse_args()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    #sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    kp_detector = onnxruntime.InferenceSession(args.onnx_file_kp, sess_options, providers=[('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 'trt_engine_cache_path': 'cache'}), 'CUDAExecutionProvider'])
    tpsmm_model = onnxruntime.InferenceSession(args.onnx_file_tpsmm, sess_options, providers=[('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 'trt_engine_cache_path': 'cache'}), 'CUDAExecutionProvider'])

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

    ort_inputs = {'in': source}
    kp_source = kp_detector.run(None, ort_inputs)[0]  # 1, 50, 2
    kp_driver_ref = None

    total_frames = driving.shape[2]
    pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
    predictions = []
    for frame_idx in range(driving.shape[2]):
        driving_frame = driving[:, :, frame_idx]
        tic = time.time()
        ort_inputs = {'in': driving_frame}
        kp_driver = kp_detector.run(None, ort_inputs)[0]
        if kp_driver_ref is None:
            kp_driver_ref = kp_driver
        kp_driver = calc_relative_kp(kp_source=kp_source, kp_driver=kp_driver, kp_driver_ref=kp_driver_ref, power=1.0)
        theta, control_points, control_params = create_transformations_params(kp_source, kp_driver)
        ort_inputs = {'in': source,
                      'theta': theta,
                      'control_points': control_points,
                      'control_params': control_params,
                      'kp_driver': kp_driver,
                      'kp_source': kp_source,
                      }
        out = tpsmm_model.run(None, ort_inputs)[0]
        toc = time.time()
        im = np.transpose(out, [0, 2, 3, 1])[0]
        #im = im.clip(0, 1)
        #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        #joinedFrame = np.concatenate((cv2_source, im, frame_face_save), axis=1)
        predictions.append(im)

        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)
    pbar.close()
    imageio.mimsave(args.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    main()
