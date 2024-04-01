import yaml
import torch
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
import argparse
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("FOMM traced model")
    parser.add_argument(
        "--output-name-kp", type=str, default="kp_detector.pt", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-fomm", type=str, default="fomm.pt", help="output name of fomm models"
    )
    parser.add_argument(
        "-f",
        "--config",
        default="config/vox-adv-256.yaml",
        type=str,
        help="yaml config file",
    )
    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox-adv-cpk.pth.tar", type=str, help="ckpt path")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not args.cpu:
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(args.ckpt)
    else:
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    if not args.cpu:
        dummy_input = dummy_input.cuda()
    kp_detector_traced = torch.jit.trace(kp_detector, dummy_input)
    torch.jit.save(kp_detector_traced, args.output_name_kp)
    logger.info("generated traced model named {}".format(args.output_name_kp))

    source = torch.randn(1, 3, 256, 256)
    driving_value = torch.randn(1, 10, 2)
    driving_jacobian = torch.randn(1, 10, 2, 2)
    source_value = torch.randn(1, 10, 2)
    source_jacobian = torch.randn(1, 10, 2, 2)
    if not args.cpu:
        source = source.cuda()
        driving_value = driving_value.cuda()
        driving_jacobian = driving_jacobian.cuda()
        source_value = source_value.cuda()
        source_jacobian = source_jacobian.cuda()
    generator_traced = torch.jit.trace(generator, (source, driving_value, driving_jacobian, source_value, source_jacobian))
    torch.jit.save(generator_traced, args.output_name_fomm)
    logger.info("generated traced model named {}".format(args.output_name_fomm))


if __name__ == "__main__":
    main()