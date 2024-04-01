import yaml
import torch
from modules.keypoint_detector import KPDetector
from modules.mapping import MappingNet
from modules.generator import OcclusionAwareSPADEGenerator
import argparse
from loguru import logger
import safetensors
import safetensors.torch 


def make_parser():
    parser = argparse.ArgumentParser("SadTalker onnx deploy")
    parser.add_argument(
        "--output-name-kp", type=str, default="kp_detector.onnx", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-generator", type=str, default="generator.onnx", help="output name of generator models"
    )
    parser.add_argument(
        "--output-name-mp", type=str, default="mapping.onnx", help="output name of mapping models"
    )
    parser.add_argument(
        "--input-kp", default="driving_frame", type=str, help="input node name of kp onnx model"
    )
    parser.add_argument(
        "--input-generator", default=["source", "driving_value", "source_value"],
        type=list,
        help="input node name of generator onnx model"
    )
    parser.add_argument(
        "--input-mp", default="3dmm", type=str, help="input node name of mp onnx model"
    )
    parser.add_argument(
        "--output-kp", default="driving_value", type=str, help="output node name of kp onnx model"
    )
    parser.add_argument(
        "--output-generator", default="output", type=str, help="output node name of generator onnx model"
    )
    parser.add_argument(
        "--output-mp", default=["yaw", "pitch", "roll", "t", "exp"], type=list, help="output node name of mp onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=13, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--config",
        default="config/facerender.yaml",
        type=str,
        help="yaml config file",
    )
    parser.add_argument("-c", "--ckpt", default="./checkpoints/SadTalker_V0.0.2_256.safetensors", type=str, help="ckpt path")
    parser.add_argument("-mc", "--mp_ckpt", default="./checkpoints/mapping_00229-model.pth.tar", type=str, help="mp ckpt path")

    return parser


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


def load_cpk_mapping(checkpoint_path, mapping=None):
    checkpoint = torch.load(checkpoint_path,  map_location=torch.device('cpu'))
    if mapping is not None:
        mapping.load_state_dict(checkpoint['mapping'])


@logger.catch
def main():
    args = make_parser().parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                             **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    
    mapping = MappingNet(**config['model_params']['mapping_params'])

    generator.to('cpu')
    kp_detector.to('cpu')
    mapping.to('cpu')
    for param in generator.parameters():
        param.requires_grad = False
    for param in kp_detector.parameters():
        param.requires_grad = False 
    for param in mapping.parameters():
        param.requires_grad = False
    
    load_cpk_facevid2vid_safetensor(args.ckpt, kp_detector=kp_detector, generator=generator)
    load_cpk_mapping(args.mp_ckpt, mapping=mapping)
    kp_detector.eval()
    generator.eval()
    mapping.eval()

    dummy_input = torch.randn(args.batch_size, 3, 256, 256)
    torch.onnx.export(
        kp_detector,
        dummy_input,
        args.output_name_kp,
        input_names=[args.input_kp],
        output_names=[args.output_kp],
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_kp))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_kp)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_kp)
        logger.info("generated simplified onnx model named {}".format(args.output_name_kp))

    source = torch.randn(args.batch_size, 3, 256, 256)
    driving_value = torch.randn(args.batch_size, 15, 3)
    source_value = torch.randn(args.batch_size, 15, 3)
    torch.onnx.export(
        generator,
        (source, driving_value, source_value),
        args.output_name_generator,
        input_names=args.input_generator,
        output_names=[args.output_generator],
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_generator))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_generator)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_generator)
        logger.info("generated simplified onnx model named {}".format(args.output_name_generator))
    
    input_3dmm = torch.randn(args.batch_size, 70, 27)
    torch.onnx.export(
        mapping,
        input_3dmm,
        args.output_name_mp,
        input_names=[args.input_mp],
        output_names=args.output_mp,
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_mp))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_mp)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_mp)
        logger.info("generated simplified onnx model named {}".format(args.output_name_mp))


if __name__ == "__main__":
    main()