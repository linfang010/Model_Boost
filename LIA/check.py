import yaml
import torch
from networks.generator import Generator
import onnx
import numpy as np
import onnxruntime


def validate_onnx(model_path):
    onnx_model = onnx.load(model_path)
    check_res = onnx.checker.check_model(onnx_model)
    print(f'lia model check:{check_res}')


def compare_onnx_torch(onnx_path, torch_path):
    generator = Generator(256, 512, 20, 1).cuda()
    weight = torch.load(torch_path, map_location=lambda storage, loc: storage)['gen']
    generator.load_state_dict(weight)
    generator.eval()

    source = np.random.random((1, 3, 256, 256)).astype('float32')
    driving_ref_motion = np.random.random((1, 20)).astype('float32')
    source_torch = torch.tensor(source).cuda()
    driving_torch = torch.tensor(driving_ref_motion).cuda()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    lia_model = onnxruntime.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider'])
    '''
    ort_inputs = {'source': np.zeros((1, 3, 256, 256), np.float32),
                  'driving': source,
                  'start': np.zeros((1, 20), np.float32)
                  }
    h_start_onnx = lia_model.run(['motion'], ort_inputs)[0]
    h_start = generator.enc.enc_motion(source_torch)
    np.testing.assert_allclose(h_start_onnx, h_start.data.cpu().numpy(), rtol=1.0, atol=1e-05)
    print("The difference of enc_motion results between ONNXRuntime and Pytorch looks good!")
    '''
    ort_inputs = {'source': source,
                  'driving': source,
                  'start': driving_ref_motion
                  }
    out_onnx = lia_model.run(None, ort_inputs)[0]
    out = generator(source_torch, source_torch, driving_torch)
    np.testing.assert_allclose(out_onnx, out.data.cpu().numpy(), rtol=1.0, atol=1e-05)
    print("The difference of generator results between ONNXRuntime and Pytorch looks good!")


if __name__ == "__main__":
    validate_onnx('generator.onnx')
    compare_onnx_torch('generator.onnx','checkpoints/vox.pt')