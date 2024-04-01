from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.transform import resize
import onnxruntime
import argparse
import imageio
from ppgan.faceutils import face_detection


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./paddle_onnx.mp4", type=str, help="generated video path"
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


def extract_bbox(image):
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        face_detector='sfd')

    frame = [image]
    predictions = detector.get_detections_for_image(np.array(frame))
    person_num = len(predictions)
    if person_num == 0:
        return np.array([])
    results = []
    face_boxs = []
    h, w, _ = image.shape
    for rect in predictions:
        bh = rect[3] - rect[1]
        bw = rect[2] - rect[0]
        cy = rect[1] + int(bh / 2)
        cx = rect[0] + int(bw / 2)
        margin = max(bh, bw)
        y1 = max(0, cy - margin)
        x1 = max(0, cx - int(0.8 * margin))
        y2 = min(h, cy + margin)
        x2 = min(w, cx + int(0.8 * margin))
        area = (y2 - y1) * (x2 - x1)
        results.append([x1, y1, x2, y2, area])
    # if a person has more than one bbox, keep the largest one
    # maybe greedy will be better?
    sorted(results, key=lambda area: area[4], reverse=True)
    results_box = [results[0]]
    for i in range(1, person_num):
        num = len(results_box)
        add_person = True
        for j in range(num):
            pre_person = results_box[j]
            iou = IOU(pre_person[0], pre_person[1], pre_person[2],
                      pre_person[3], pre_person[4], results[i][0],
                      results[i][1], results[i][2], results[i][3],
                      results[i][4])
            if iou > 0.5:
                add_person = False
                break
        if add_person:
            results_box.append(results[i])
    boxes = np.array(results_box)
    return boxes


def IOU(ax1, ay1, ax2, ay2, sa, bx1, by1, bx2, by2, sb):
    #sa = abs((ax2 - ax1) * (ay2 - ay1))
    #sb = abs((bx2 - bx1) * (by2 - by1))
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    w = x2 - x1
    h = y2 - y1
    if w < 0 or h < 0:
        return 0.0
    else:
        return 1.0 * w * h / (sa + sb - w * h)
    

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

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    #sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    kp_detector = onnxruntime.InferenceSession(args.onnx_file_kp, sess_options, providers=[('TensorrtExecutionProvider',{'trt_fp16_enable': True,}), 'CUDAExecutionProvider'])
    fomm_model = onnxruntime.InferenceSession(args.onnx_file_fomm, sess_options, providers=[('TensorrtExecutionProvider',{'trt_fp16_enable': True,}), 'CUDAExecutionProvider'])

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
    bboxes = extract_bbox(source_image.copy())
    print(str(len(bboxes)) + " persons have been detected")
    rec = bboxes[0]
    face_image = source_image.copy()[rec[1]:rec[3], rec[0]:rec[2]]
    face_image = cv2.resize(face_image, (256, 256)) / 255.0
    source = np.transpose(face_image[np.newaxis].astype(np.float32), (0, 3, 1, 2))

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
        im = np.transpose(out, [0, 2, 3, 1])[0] * 255.0
        predictions.append(im)
        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)
    pbar.close()
    out_frame = []
    for i in range(len(driving_video)):
        frame = source_image.copy()
        x1, y1, x2, y2, _ = rec
        out = predictions[i]
        out = cv2.resize(out.astype(np.uint8), (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = out
        out_frame.append(frame)

    imageio.mimsave(args.output, [frame for frame in out_frame], fps=fps)


if __name__ == "__main__":
    main()
