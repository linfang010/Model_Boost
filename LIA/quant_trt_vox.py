import onnxruntime
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, CalibrationMethod, write_calibration_table
import imageio
import cv2
import os
import random


class LiaDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, img_path: str, video_path: str):
        self.enum_data = None
        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(model_path, sess_options, providers=['CUDAExecutionProvider'])
        self.input_name1 = session.get_inputs()[0].name
        self.input_name2 = session.get_inputs()[1].name
        self.input_name3 = session.get_inputs()[2].name
        self.input_data_list = []
        source = self.read_img(img_path)
        # driving = self.vid_preprocessing(video_path)
        driving = self.read_img(video_path)
        ort_inputs = {'source': np.zeros((1, 3, 256, 256), np.float32),
                      'driving': driving,
                      'start': np.zeros((1, 20), np.float32)
                      }
        start = session.run(['motion'], ort_inputs)[0]
        self.input_data_list.append((source, driving, start))
    
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name1: input_data[0], self.input_name2: input_data[1], self.input_name3: input_data[2]} for input_data in self.input_data_list]
            )
        return next(self.enum_data, None)

    def read_img(self, path):
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # som images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = (cv2.resize(img, (256, 256)) / 255.0 - 0.5) * 2.0  # [-1, 1]
        img = np.transpose(img[np.newaxis].astype(np.float32), (0, 3, 1, 2))
        return img
    
    def vid_preprocessing(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        # video_fps = cap.get(cv2.CAP_PROP_FPS)

        video = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array([
            cv2.resize(frame, (256, 256)) for frame in video
        ])

        vid = np.transpose(video[np.newaxis].astype(np.float32), (0, 4, 1, 2, 3))
        # fps = video_fps
        vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

        return vid_norm[:, :, 0]


if __name__ == '__main__':
    model_path = './generator-infer.onnx'
    augmented_model_path = "./augmented_model.onnx"
    vox_path = '../../unzippedFaces'
    calib_list = []
    op_types_to_quantize = ['Reshape', 'Mul', 'Add']
    # op_types_to_quantize = []

    if os.path.exists(vox_path):
        celeb_list = os.listdir(vox_path)
        source_list = random.sample(celeb_list, k=1000)
        for temp in source_list:
            source_path = vox_path + '/' + temp + '/1.6'
            source_path_list = os.listdir(source_path)
            source_temp = random.choice(source_path_list)
            source_final_path = source_path + '/' + source_temp
            source_img_list = os.listdir(source_final_path)
            source_img = random.choice(source_img_list)

            temp1 = random.choice(celeb_list)
            driving_path = vox_path + '/' + temp1 + '/1.6'
            driving_path_list = os.listdir(driving_path)
            driving_temp = random.choice(driving_path_list)
            driving_final_path = driving_path + '/' + driving_temp
            driving_img_list = os.listdir(driving_final_path)
            driving_img = random.choice(driving_img_list)

            final = (source_final_path + '/' + source_img, driving_final_path + '/' + driving_img)
            calib_list.append(final)

    # Generate INT8 calibration cache
    print("Calibration starts ...")
    calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Percentile)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 

    for calib in calib_list:
        img_path = calib[0]
        video_path = calib[1]
        data_reader = LiaDataReader(model_path, img_path, video_path)
        calibrator.collect_data(data_reader)
    
    compute_range = calibrator.compute_range()
    write_calibration_table(compute_range)
    print("Calibration is done. Calibration cache is saved to calibration.json")
