import argparse
from omegaconf import OmegaConf
from trainer import RSDTrainer as TrainerDistill

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./saved_logs",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--exp_name",
            type=str,
            default="rsd_distill_on_imagenet_x4",
            help="Experiment name",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="Resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/RSD.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--steps",
            type=int,
            default=15,
            help="Hyper-parameters of diffusion steps",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    configs.diffusion.params.steps = args.steps

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = TrainerDistill(configs)
    trainer.train()
