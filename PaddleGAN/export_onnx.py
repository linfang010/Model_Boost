import yaml
import paddle
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator
from ppgan.modules.keypoint_detector import KPDetector
import argparse
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("PaddleGAN FOM onnx deploy")
    parser.add_argument(
        "--output-name-kp", type=str, default="paddle_kp_detector", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-fomm", type=str, default="paddle_generator", help="output name of fomm models"
    )
    parser.add_argument(
        "--input-kp", default="driving_frame", type=str, help="input node name of kp onnx model"
    )
    parser.add_argument(
        "--input-fomm", default=["source", "driving_value", "driving_jacobian", "source_value", "source_jacobian"],
        type=list,
        help="input node name of tpsmm onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--config",
        default="configs/firstorder_vox_256.yaml",
        type=str,
        help="yaml config file",
    )
    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox-cpk.pdparams", type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # paddle.set_device('cpu')

    generator = OcclusionAwareGenerator(**config['model']['generator']['generator_cfg'],
                                        **config['model']['common_params'])

    kp_detector = KPDetector(**config['model']['generator']['kp_detector_cfg'],
                             **config['model']['common_params'])

    checkpoint = paddle.load(args.ckpt)

    generator.set_state_dict(checkpoint['generator'])
    kp_detector.set_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()
    
    x_spec = paddle.static.InputSpec([args.batch_size, 3, 256, 256], 'float32', args.input_kp)
    paddle.onnx.export(
        kp_detector,
        args.output_name_kp,
        input_spec=[x_spec],
        opset_version=args.opset
    )

    logger.info("generated onnx model named {}".format(args.output_name_kp))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_kp+'.onnx')
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_kp+'.onnx')
        logger.info("generated simplified onnx model named {}".format(args.output_name_kp))

    source = paddle.static.InputSpec([1, 3, 256, 256], 'float32', args.input_fomm[0])
    driving_value = paddle.static.InputSpec([1, 10, 2], 'float32', args.input_fomm[1])
    driving_jacobian = paddle.static.InputSpec([1, 10, 2, 2], 'float32', args.input_fomm[2])
    source_value = paddle.static.InputSpec([1, 10, 2], 'float32', args.input_fomm[3])
    source_jacobian = paddle.static.InputSpec([1, 10, 2, 2], 'float32', args.input_fomm[4])

    paddle.onnx.export(
        generator,
        args.output_name_fomm,
        input_spec=[source, driving_value, driving_jacobian, source_value, source_jacobian],
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_fomm))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_fomm+'.onnx')
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_fomm+'.onnx')
        logger.info("generated simplified onnx model named {}".format(args.output_name_fomm))


if __name__ == "__main__":
    main()

# python export_onnx.py --output-name-kp paddle_kp_detector --output-name-fomm paddle_generator --config configs/firstorder_vox_256.yaml --ckpt ./checkpoints/vox-cpk.pdparams
