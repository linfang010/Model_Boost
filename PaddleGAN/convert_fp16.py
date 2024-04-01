import yaml
import paddle
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator
from ppgan.modules.keypoint_detector import KPDetector
import argparse
from loguru import logger
from paddle.fluid.contrib.slim.quantization.post_training_quantization import WeightQuantization


def make_parser():
    parser = argparse.ArgumentParser("PaddleGAN FOM onnx deploy")
    parser.add_argument(
        "--output-name-kp", type=str, default="paddle_kp_detector", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-fomm", type=str, default="paddle_generator", help="output name of fomm models"
    )
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

    paddle.jit.save(generator, 'generator')
    paddle.jit.save(kp_detector, 'kp_detector')

    generator_wq = WeightQuantization(model_dir='kp_detector', model_filename='model.pdmodel', params_filename='model.pdparams')
    generator_wq.convert_weight_to_fp16(args.output_name_kp)
    logger.info("generated fp16 model in {}".format(args.output_name_kp))

    generator_wq = WeightQuantization(model_dir='generator', model_filename='model.pdmodel', params_filename='model.pdparams')
    generator_wq.convert_weight_to_fp16(args.output_name_fomm)
    logger.info("generated fp16 model in {}".format(args.output_name_fomm))


if __name__ == "__main__":
    main()

# python export_onnx.py --output-name-kp paddle_kp_detector --output-name-fomm paddle_generator --config configs/firstorder_vox_256.yaml --ckpt ./checkpoints/vox-cpk.pdparams
