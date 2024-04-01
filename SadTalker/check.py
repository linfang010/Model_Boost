import yaml
import torch
from src.facerender.modules.keypoint_detector import KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
import onnx
import numpy as np
import onnxruntime
import safetensors
import safetensors.torch 


def validate_onnx(model_path):
    onnx_model = onnx.load(model_path)
    check_res = onnx.checker.check_model(onnx_model)
    print(f'sadtalker model check:{check_res}')


def load_cpk_facevid2vid_safetensor(checkpoint_path, generator=None, kp_detector=None):

    checkpoint = safetensors.torch.load_file(checkpoint_path)

    if generator is not None:
        x_generator = {}
        for k, v in checkpoint.items():
            if 'generator' in k:
                x_generator[k.replace('generator.', '')] = v
        generator.load_state_dict(x_generator)
    if kp_detector is not None:
        x_generator = {}
        for k, v in checkpoint.items():
            if 'kp_extractor' in k:
                x_generator[k.replace('kp_extractor.', '')] = v
        kp_detector.load_state_dict(x_generator)


def compare_onnx_torch(kp_onnx_path, generator_onnx_path, safetensor_path, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                             **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    
    generator.to('cuda')
    kp_detector.to('cuda')
    for param in generator.parameters():
        param.requires_grad = False
    for param in kp_detector.parameters():
        param.requires_grad = False 
    load_cpk_facevid2vid_safetensor(safetensor_path, kp_detector=kp_detector, generator=generator)
    kp_detector.eval()
    generator.eval()

    source_image = np.random.random((2, 3, 256, 256)).astype('float32')
    source_value = np.random.random((2, 15, 3)).astype('float32')
    source_image_torch = torch.tensor(source_image).cuda()
    source_value_torch = torch.tensor(source_value).cuda()
    kp_source = {'value': source_value_torch}

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    kp_detector_onnx = onnxruntime.InferenceSession(kp_onnx_path, sess_options, providers=['CUDAExecutionProvider'])
    generator_onnx = onnxruntime.InferenceSession(generator_onnx_path, sess_options, providers=['CUDAExecutionProvider'])

    ort_inputs = {kp_detector_onnx.get_inputs()[0].name: source_image}
    out_onnx = kp_detector_onnx.run(None, ort_inputs)[0]
    out = kp_detector(source_image_torch)
    np.testing.assert_allclose(out_onnx, out['value'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    print("The difference of kp_detector results between ONNXRuntime and Pytorch looks good!")

    ort_inputs = {'source': source_image,
                  'driving_value': source_value,
                  'source_value': source_value
                  }
    out_onnx = generator_onnx.run(None, ort_inputs)[0]
    out = generator(source_image_torch, kp_source, kp_source)
    np.testing.assert_allclose(out_onnx, out['prediction'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    print("The difference of generator results between ONNXRuntime and Pytorch looks good!")


def load_cpk_mapping(checkpoint_path, mapping=None):
    checkpoint = torch.load(checkpoint_path,  map_location=torch.device('cuda'))
    if mapping is not None:
        mapping.load_state_dict(checkpoint['mapping'])


def compare_mapping(onnx_path, checkpoint_path, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mapping = MappingNet(**config['model_params']['mapping_params'])
    mapping.to('cuda')
    for param in mapping.parameters():
        param.requires_grad = False
    load_cpk_mapping(checkpoint_path, mapping=mapping)
    mapping.eval()

    input_3dmm = np.random.random((2, 70, 27)).astype('float32')
    input_3dmm_torch = torch.tensor(input_3dmm).cuda()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    mapping_onnx = onnxruntime.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider'])

    ort_inputs = {mapping_onnx.get_inputs()[0].name: input_3dmm}
    yaw, pitch, roll, t, exp = mapping_onnx.run(None, ort_inputs)
    out = mapping(input_3dmm_torch)
    np.testing.assert_allclose(yaw, out['yaw'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(pitch, out['pitch'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(roll, out['roll'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(t, out['t'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    np.testing.assert_allclose(exp, out['exp'].data.cpu().numpy(), rtol=1.0, atol=1e-05)
    print("The difference of mapping results between ONNXRuntime and Pytorch looks good!")


if __name__ == "__main__":
    validate_onnx('generator.onnx')
    validate_onnx('kp_detector.onnx')
    validate_onnx('mapping.onnx')
    compare_onnx_torch('kp_detector.onnx', 'generator.onnx', 'checkpoints/SadTalker_V0.0.2_256.safetensors', 'src/config/facerender.yaml')
    compare_mapping('mapping.onnx', 'checkpoints/mapping_00229-model.pth.tar', 'src/config/facerender.yaml')