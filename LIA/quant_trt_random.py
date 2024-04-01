import onnx
import onnxruntime
import numpy as np
import json
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QDQQuantizer


class LiaDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, data_size: int):
        self.enum_data = None
        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(model_path, sess_options, providers=['CUDAExecutionProvider'])
        self.input_name1 = session.get_inputs()[0].name
        self.input_name2 = session.get_inputs()[1].name
        self.input_name3 = session.get_inputs()[2].name
        self.data_size = data_size
        self.input_data_list = []
        for i in range(data_size):
            source = (np.random.random((1, 3, 256, 256)).astype('float32') - 0.5) * 2  # [-1, 1]
            driving = (np.random.random((1, 3, 256, 256)).astype('float32') - 0.5) * 2  # [-1 ,1]
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


def get_op_nodes_not_followed_by_specific_op(model, op1, op2):
    op1_nodes = []
    op2_nodes = []
    selected_op1_nodes = []
    not_selected_op1_nodes = []

    for node in model.graph.node:
        if node.op_type == op1:
            op1_nodes.append(node)
        if node.op_type == op2:
            op2_nodes.append(node)

    for op1_node in op1_nodes:
        for op2_node in op2_nodes:
            if op1_node.output == op2_node.input:
                selected_op1_nodes.append(op1_node.name)
        if op1_node.name not in selected_op1_nodes:
            not_selected_op1_nodes.append(op1_node.name)

    return not_selected_op1_nodes


if __name__ == '__main__':
    model_path = './generator-infer.onnx'
    augmented_model_path = "./augmented_model.onnx"
    qdq_model_path = "./qdq_model.onnx"
    calib_num = 500
    op_types_to_quantize = ['Reshape', 'Mul', 'Add', 'Conv', 'Slice', 'Pad', 'Unsqueeze', 'Where', 'LeakyRelu', 'Sub', 'Div', 'Cast', 'Gather', 'Gemm', 'Expand', 
                            'GatherElements', 'Greater', 'Less', 'ReduceSum', 'Pow', 'Sqrt', 'Floor', 'Transpose', 'ConvTranspose', 'Sigmoid', 'Tanh', 'MatMul',
                            'Flatten', 'Tile']
    # op_types_to_quantize = []

    # Generate INT8 calibration cache
    print("Calibration starts ...")
    calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Percentile)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 

    for i in range(calib_num):
        data_reader = LiaDataReader(model_path, 1)
        calibrator.collect_data(data_reader)
    
    compute_range = calibrator.compute_range()
    write_calibration_table(compute_range)
    print("Calibration is done. Calibration cache is saved to calibration.json")
    '''
    # Generate QDQ model
    mode = QuantizationMode.QLinearOps

    model = onnx.load_model(Path(model_path), False)

    # In TRT, it recommended to add QDQ pair to inputs of Add node followed by ReduceMean node.
    nodes_to_exclude = get_op_nodes_not_followed_by_specific_op(model, "Add", "ReduceMean")

    quantizer = QDQQuantizer(
        model,
        False, #per_channel
        False, #reduce_range
        mode,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range,
        [], #nodes_to_quantize
        nodes_to_exclude,
        op_types_to_quantize,
        {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'OpTypesToExcludeOutputQuantization': op_types_to_quantize, 'DedicatedQDQPair': True, 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} }) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(qdq_model_path, False)
    print("QDQ model is saved to ", qdq_model_path)
    '''
