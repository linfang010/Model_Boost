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

'''
此函数需要放到原始的PaddleGAN-develop目录中运行
'''
def compare_onnx_torch(kp_path, fomm_path):
    
    with open('configs/firstorder_vox_256.yaml') as f:
        config = yaml.safe_load(f)

    # paddle.set_device('cpu')

    generator = OcclusionAwareGenerator(**config['model']['generator']['generator_cfg'],
                                        **config['model']['common_params'],
                                         inference=True)

    kp_detector = KPDetector(**config['model']['generator']['kp_detector_cfg'],
                             **config['model']['common_params'])

    checkpoint = paddle.load('./checkpoints/vox-cpk.pdparams')

    generator.set_state_dict(checkpoint['generator'])
    kp_detector.set_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()

    source = np.random.random((1, 3, 256, 256)).astype('float32')
    driving_value = np.random.random((1, 10, 2)).astype('float32')
    driving_jacobian = np.random.random((1, 10, 2, 2)).astype('float32')
    source_value = np.random.random((1, 10, 2)).astype('float32')
    source_jacobian = np.random.random((1, 10, 2, 2)).astype('float32')

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    kp_detector_onnx = onnxruntime.InferenceSession(kp_path, sess_options, providers=['CUDAExecutionProvider'])
    fomm_model = onnxruntime.InferenceSession(fomm_path, sess_options, providers=['CUDAExecutionProvider'])
    
    ort_inputs = {kp_detector_onnx.get_inputs()[0].name: source}
    kp_source_onnx, kp_source_jacobian_onxx = kp_detector_onnx.run(None, ort_inputs)
    paddle_source = paddle.to_tensor(source)
    kp_source_paddle = kp_detector(paddle_source)
    np.testing.assert_allclose(kp_source_onnx, kp_source_paddle['value'].numpy(), rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(kp_source_jacobian_onxx, kp_source_paddle['jacobian'].numpy(), rtol=1.0, atol=1e-05)
    print("The difference of kp_detector results between ONNXRuntime and Paddle looks good!")

    ort_inputs = {fomm_model.get_inputs()[0].name: source,
                  fomm_model.get_inputs()[1].name: driving_value,
                  fomm_model.get_inputs()[2].name: driving_jacobian,
                  fomm_model.get_inputs()[3].name: source_value,
                  fomm_model.get_inputs()[4].name: source_jacobian,
                      }
    out_onnx = fomm_model.run(None, ort_inputs)[0]
    kp_source = {}
    kp_source["value"] = paddle.to_tensor(source_value)
    kp_source["jacobian"] = paddle.to_tensor(source_jacobian)
    kp_driving = {}
    kp_driving["value"] = paddle.to_tensor(driving_value)
    kp_driving["jacobian"] = paddle.to_tensor(driving_jacobian)
    out = generator(paddle_source, kp_source=kp_source, kp_driving=kp_driving)
    np.testing.assert_allclose(out_onnx, out['prediction'].numpy(), rtol=1.0, atol=1e-05)
    print("The difference of fomm results between ONNXRuntime and Paddle looks good!")


def compare_onnx_fp16_fp32(kp_path, fomm_path):
    source = np.random.random((1, 3, 256, 256)).astype('float32')
    driving_value = np.random.random((1, 10, 2)).astype('float32')
    driving_jacobian = np.random.random((1, 10, 2, 2)).astype('float32')
    source_value = np.random.random((1, 10, 2)).astype('float32')
    source_jacobian = np.random.random((1, 10, 2, 2)).astype('float32')

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    kp_detector = onnxruntime.InferenceSession(kp_path+'.onnx', sess_options, providers=['CUDAExecutionProvider'])
    fomm_model = onnxruntime.InferenceSession(fomm_path+'.onnx', sess_options, providers=['CUDAExecutionProvider'])
    kp_detector_fp16 = onnxruntime.InferenceSession(kp_path+'_fp16.onnx', sess_options, providers=['CUDAExecutionProvider'])
    fomm_model_fp16 = onnxruntime.InferenceSession(fomm_path+'_fp16.onnx', sess_options, providers=['CUDAExecutionProvider'])

    ort_inputs = {kp_detector.get_inputs()[0].name: source}
    kp_source, kp_source_jacobian = kp_detector.run(None, ort_inputs)
    ort_inputs = {kp_detector_fp16.get_inputs()[0].name: source}
    kp_source_fp16, kp_source_jacobian_fp16 = kp_detector_fp16.run(None, ort_inputs)
    np.testing.assert_allclose(kp_source, kp_source_fp16, rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(kp_source_jacobian, kp_source_jacobian_fp16, rtol=1.0, atol=1e-05)
    print("The difference of kp_detector results between fp32 and fp16 looks good!")

    ort_inputs = {fomm_model.get_inputs()[0].name: source,
                  fomm_model.get_inputs()[1].name: driving_value,
                  fomm_model.get_inputs()[2].name: driving_jacobian,
                  fomm_model.get_inputs()[3].name: source_value,
                  fomm_model.get_inputs()[4].name: source_jacobian,
                      }
    out = fomm_model.run(None, ort_inputs)[0]
    ort_inputs = {fomm_model_fp16.get_inputs()[0].name: source,
                  fomm_model_fp16.get_inputs()[1].name: driving_value,
                  fomm_model_fp16.get_inputs()[2].name: driving_jacobian,
                  fomm_model_fp16.get_inputs()[3].name: source_value,
                  fomm_model_fp16.get_inputs()[4].name: source_jacobian,
                      }
    out_fp16 = fomm_model_fp16.run(None, ort_inputs)[0]
    np.testing.assert_allclose(out, out_fp16, rtol=1.0, atol=1e-05)
    print("The difference of fomm results between fp32 and fp16 looks good!")



if __name__ == "__main__":
    # validate_onnx('paddle_kp_detector.onnx', 'paddle_generator.onnx')
    #compare_onnx_torch('paddle_kp_detector.onnx', 'paddle_generator.onnx')
    compare_onnx_fp16_fp32('paddle_kp_detector','paddle_generator')
    