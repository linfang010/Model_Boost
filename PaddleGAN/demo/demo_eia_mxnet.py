from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.transform import resize
import argparse
import imageio
import mxnet as mx
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from scipy.spatial import ConvexHull


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./mxnet_onnx.mp4", type=str, help="generated video path"
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
        source_area = ConvexHull(kp_source[0].asnumpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0].asnumpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    
    if use_relative_movement:
        kp_value_diff = (kp_driving - kp_driving_initial)
        kp_value_diff *= movement_scale
        kp_new = kp_value_diff + kp_source
        if use_relative_jacobian:
            jacobian_diff = mx.np.matmul(kp_driving_jacobian, mx.np.linalg.inv(kp_driving_initial_jacobian))
            kp_new_jacobian = mx.np.matmul(jacobian_diff, kp_source_jacobian)
    
    return kp_new, kp_new_jacobian
    

def read_img(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # som images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def main():
    args = make_parser().parse_args()

    ctx = mx.gpu()
    sym, arg_params, aux_params = import_model(args.onnx_file_kp)
    kp_detector = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    kp_detector.bind(for_training=False, data_shapes=[('driving_frame', (1, 3, 256, 256))], label_shapes=kp_detector._label_shapes)
    kp_detector.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    sym, arg_params, aux_params = import_model(args.onnx_file_fomm)
    fomm_model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    fomm_model.bind(for_training=False,
                    data_shapes=[('source', (1, 3, 256, 256)), ('driving_value', (1, 10, 2)), ('driving_jacobian', (1, 10, 2, 2)),
                                 ('source_value', (1, 10, 2)), ('source_jacobian', (1, 10, 2, 2))],
                    label_shapes=kp_detector._label_shapes)

    source_image = read_img(args.source)
    reader = imageio.get_reader(args.driving)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [cv2.resize(frame, (256, 256)) / 255.0 for frame in driving_video]
    driving = np.transpose(np.array(driving_video)[np.newaxis].astype(np.float32), (0, 4, 1, 2, 3))
    face_image = source_image.copy()
    face_image = cv2.resize(face_image, (256, 256)) / 255.0
    source = np.transpose(face_image[np.newaxis].astype(np.float32), (0, 3, 1, 2))

    kp_detector.predict(mx.nd.array(source))
    kp_source, kp_source_jacobian = kp_detector.get_outputs()[0]
    kp_detector.predict(mx.nd.array(driving[:, :, 0]))
    kp_driving_initial, kp_driving_initial_jacobian = kp_detector.get_outputs()[0]

    total_frames = driving.shape[2]
    pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
    predictions = []
    for frame_idx in range(driving.shape[2]):
        driving_frame = driving[:, :, frame_idx]
        tic = time.time()
        kp_detector.predict(mx.nd.array(driving_frame))
        kp_driving, kp_driving_jacobian = kp_detector.get_outputs()[0]
        kp_norm, kp_norm_jacobian = normalize_kp(kp_source, kp_source_jacobian, kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian,
                                                 adapt_movement_scale=args.adapt_scale, use_relative_movement=args.relative, use_relative_jacobian=args.relative)
        data = {'source': mx.nd.array(source),
                'driving_value': mx.nd.array(kp_norm),
                'driving_jacobian': mx.nd.array(kp_norm_jacobian),
                'source_value': mx.nd.array(kp_source),
                'source_jacobian': mx.nd.array(kp_source_jacobian),
                }
        nd_iter = mx.io.NDArrayIter(data=data)
        fomm_model.predict(nd_iter)
        out = fomm_model.get_outputs()[0]
        toc = time.time()
        im = np.transpose(out.asnumpy(), [0, 2, 3, 1])[0]
        predictions.append(im)
        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)
    pbar.close()

    imageio.mimsave(args.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    main()
