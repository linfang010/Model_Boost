import onnx
from onnxconverter_common import float16


def convert_fp16(onnx_path, save_path):
    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, save_path)


if __name__ == '__main__':
    convert_fp16('onnx_models/generator.onnx', 'onnx_models/generator_fp16.onnx')
    convert_fp16('onnx_models/kp_detector.onnx', 'onnx_models/kp_detector_fp16.onnx')
    convert_fp16('onnx_models/mapping.onnx', 'onnx_models/mapping_fp16.onnx')