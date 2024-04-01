from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.transform import resize
import onnxruntime
import argparse
import imageio
from PIL import Image


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./lia_onnx.mp4", type=str, help="generated video path"
    )

    parser.add_argument("-c", "--onnx-file-lia", type=str, help="lia onnx model path")

    return parser


def img_preprocessing(img_path, size=256):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img[np.newaxis].astype(np.float32), (0, 3, 1, 2))  # 1 x 3 x 256 x 256
    imgs_norm = (img / 255.0 - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path, size=256):
    cap = cv2.VideoCapture(vid_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array([
        cv2.resize(frame, (size, size)) for frame in video
    ])

    vid = np.transpose(video[np.newaxis].astype(np.float32), (0, 4, 1, 2, 3))
    fps = video_fps
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def main():
    args = make_parser().parse_args()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    #sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    generator = onnxruntime.InferenceSession(args.onnx_file_lia, sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

    source = img_preprocessing(args.source)
    driving, fps = vid_preprocessing(args.driving)

    ort_inputs = {'in_src': np.zeros((1, 3, 256, 256), np.float32),
                  'in_drv': driving[:, :, 0],
                  'in_drv_start_motion': np.zeros((1, 20), np.float32),
                  'in_power': np.zeros((1,), np.float32)}
    driving_ref_motion = generator.run(['out_drv_motion'], ort_inputs)[0]

    total_frames = driving.shape[2]
    pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
    predictions = []
    for frame_idx in range(driving.shape[2]):
        driving_frame = driving[:, :, frame_idx]
        tic = time.time()
        ort_inputs = {'in_src': source,
                      'in_drv': driving_frame,
                      'in_drv_start_motion': driving_ref_motion,
                      'in_power': np.array([1.0], np.float32)
                      }
        out = generator.run(['out'], ort_inputs)[0]
        toc = time.time()
        im = np.transpose(out, [0, 2, 3, 1])[0]
        predictions.append(im)
        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)
    pbar.close()
    vid = np.array(predictions)
    vid = vid.clip(-1, 1)
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).astype(np.uint8)

    imageio.mimsave(args.output, [frame for frame in vid], fps=fps)


if __name__ == "__main__":
    main()
