# import yaml
import torch
from networks.generator import Generator
import argparse
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("LIA onnx deploy")
    parser.add_argument(
        "--output-name-lia", type=str, default="generator.onnx", help="output name of lia models"
    )
    parser.add_argument(
        "--input-lia", default=["source", "driving", "start"],
        type=list,
        help="input node name of lia onnx model"
    )
    parser.add_argument(
        "--output-lia", default=["out", "motion"], type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=13, type=int, help="onnx opset version"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox.pt", type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    generator = Generator(256, 512, 20, 1)
    weight = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['gen']
    generator.load_state_dict(weight)
    generator.eval()

    source = torch.randn(1, 3, 256, 256)
    driving = torch.randn(1, 3, 256, 256)
    start = torch.randn(1, 20)

    torch.onnx.export(
        generator,
        (source, driving, start),
        args.output_name_lia,
        input_names=args.input_lia,
        output_names=args.output_lia,
        opset_version=args.opset
    )

    logger.info("generated onnx model named {}".format(args.output_name_lia))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_lia)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_lia)
        logger.info("generated simplified onnx model named {}".format(args.output_name_lia))


if __name__ == "__main__":
    main()

# python export_onnx.py --output-name-kp kp_detector.onnx --output-name-fomm fomm.onnx --config config/vox-adv-256.yaml --ckpt ./checkpoints/vox-adv-cpk.pth.tar
