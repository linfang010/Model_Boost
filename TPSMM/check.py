import yaml
import paddle
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator
from ppgan.modules.keypoint_detector import KPDetector
import onnx
import numpy as np
import onnxruntime


def validate_onnx(kp_path, fomm_path):
    onnx_model = onnx.load(kp_path)
    check_res = onnx.checker.check_model(onnx_model)
    print(f'kp_detector check:{check_res}')
    onnx_model = onnx.load(fomm_path)
    check_res = onnx.checker.check_model(onnx_model)
    print(f'fomm_path check:{check_res}')


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


def compare_onnx_fp16_fp32(kp_path, fomm_path):
    source = np.random.random((1, 3, 256, 256)).astype('float32')
    driving_value = np.random.random((1, 50, 2)).astype('float32')
    source_value = np.random.random((1, 50, 2)).astype('float32')

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    kp_detector = onnxruntime.InferenceSession(kp_path+'.onnx', sess_options, providers=['CUDAExecutionProvider'])
    tpsmm_model = onnxruntime.InferenceSession(fomm_path+'.onnx', sess_options, providers=['CUDAExecutionProvider'])
    kp_detector_fp16 = onnxruntime.InferenceSession(kp_path+'_fp16.onnx', sess_options, providers=['CUDAExecutionProvider'])
    tpsmm_model_fp16 = onnxruntime.InferenceSession(fomm_path+'_fp16.onnx', sess_options, providers=['CUDAExecutionProvider'])

    ort_inputs = {'in': source}
    kp_source = kp_detector.run(None, ort_inputs)
    kp_source_fp16 = kp_detector_fp16.run(None, ort_inputs)
    np.testing.assert_allclose(kp_source, kp_source_fp16, rtol=1.0, atol=1e-05)
    print("The difference of kp_detector results between fp32 and fp16 looks good!")

    theta, control_points, control_params = create_transformations_params(source_value, driving_value)
    ort_inputs = {'in': source,
                  'theta': theta,
                  'control_points': control_points,
                  'control_params': control_params,
                  'kp_driver': driving_value,
                  'kp_source': source_value,
                  }
    out = tpsmm_model.run(None, ort_inputs)[0]
    out_fp16 = tpsmm_model_fp16.run(None, ort_inputs)[0]
    np.testing.assert_allclose(out, out_fp16, rtol=1.0, atol=1e-05)
    print("The difference of fomm results between fp32 and fp16 looks good!")



if __name__ == "__main__":
    # validate_onnx('paddle_kp_detector.onnx', 'paddle_generator.onnx')
    #compare_onnx_torch('paddle_kp_detector.onnx', 'paddle_generator.onnx')
    compare_onnx_fp16_fp32('kp_detector','generator')
    