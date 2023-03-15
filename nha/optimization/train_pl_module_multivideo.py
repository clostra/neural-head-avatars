from tqdm.auto import tqdm

from nha.util.log import get_logger

import pytorch_lightning as pl
from pathlib import Path
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from configargparse import ArgumentParser as ConfigArgumentParser

logger = get_logger(__name__)


def train_pl_module_multivideo(optimizer_module, data_module, args=None):
    """
    optimizes an instance of the given optimization module on an instance of a given data_module. Takes arguments
    either from CLI or from 'args'

    :param optimizer_module:
    :param data_module:
    :param args: list similar to sys.argv to manually insert args to parse from
    :return:
    """
    # creating argument parser
    parser = ArgumentParser()
    parser = optimizer_module.add_argparse_args(parser)
    parser = data_module.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument('--config', required=True, is_config_file=True)
    parser.add_argument("--checkpoint_file", type=str, required=False, default="",
                        help="checkpoint to load model from")
    parser.add_argument("--video_selection_path", type=str, required=True,
                    help="json file to grab the video names from")

    args = parser.parse_args() if args is None else parser.parse_args(args)
    args_dict = vars(args)

    print(f"Start Model training with the following configuration: \n {parser.format_values()}")

    # init datamodule
    data = data_module(**args_dict)
    data.setup()

    if args.checkpoint_file:
        model = optimizer_module.load_from_checkpoint(args.checkpoint_file, strict=True, **args_dict)
    else:

        model = optimizer_module(
            num_videos=len(data.video_paths),
            frames_per_video=[len(data.get_frames(video_name)) for video_name in data.video_names],
            **args_dict
        )

    stages = ["offset", "texture", "joint"]
    stage_jumps = [args_dict["epochs_offset"], args_dict["epochs_offset"] + args_dict["epochs_texture"],
                   args_dict["epochs_offset"] + args_dict["epochs_texture"] + args_dict["epochs_joint"]]

    experiment_logger = TensorBoardLogger(args_dict["default_root_dir"],
                                          name="lightning_logs")
    log_dir = Path(experiment_logger.log_dir)

    for i, stage in enumerate(stages):
        current_epoch = torch.load(args_dict["checkpoint_file"])["epoch"] if args_dict["checkpoint_file"] else 0
        if current_epoch < stage_jumps[i]:
            logger.info(f"Running the {stage}-optimization stage.")

            ckpt_file = args_dict["checkpoint_file"] if args_dict["checkpoint_file"] else None
            trainer = pl.Trainer.from_argparse_args(args, callbacks=model.callbacks,
                                                    resume_from_checkpoint=ckpt_file,
                                                    max_epochs=stage_jumps[i],
                                                    logger=experiment_logger)

            # training
            trainer.fit(model,
                        train_dataloader=data.train_dataloader(batch_size=data._train_batch[i]),
                        val_dataloaders=data.val_dataloader(batch_size=data._val_batch[i]))

            ckpt_path = Path(trainer.log_dir) / "checkpoints" / (stage + "_optim.ckpt")
            trainer.save_checkpoint(ckpt_path)
            ckpt_path = Path(trainer.log_dir) / "checkpoints" / "last.ckpt"
            args_dict["checkpoint_file"] = ckpt_path
