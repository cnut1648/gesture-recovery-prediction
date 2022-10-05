from typing import List
import os

import pytorch_lightning as pl
import rich.syntax
import rich.tree
import wandb
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def fail_on_missing(cfg: DictConfig) -> None:
    if isinstance(cfg, ListConfig):
        for x in cfg:
            fail_on_missing(x)
    elif isinstance(cfg, DictConfig):
        for _, v in cfg.items():
            fail_on_missing(v)


@rank_zero_only
def pretty_print(
        cfg: DictConfig,
        fields=(
            "trainer",
            "callback",
            "logger",
            "data",
            "module",
        )
):
    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # others defined in root
    others = tree.add("others", style=style, guide_style=style)
    for var, val in OmegaConf.to_container(cfg, resolve=True).items():
        if not var.startswith("_") and var not in fields:
            others.add(f"{var}: {val}")

    rich.print(tree)


def touch(cfg: DictConfig) -> DictConfig:
    """handle special case of the cfg"""
    if cfg.test_from_run:
            cfg = restore(cfg)
    with open_dict(cfg):
        if cfg.test_from_run:
            del cfg.logger
            del cfg.callback
    
    # if os.environ["CUDA_VISIBLE_DEVICES"] and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
    #     cfg.trainer.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    #     cfg.trainer.accelerator = "ddp"

    fail_on_missing(cfg)

    if cfg.debug:
        # turn off wandb if present
        if cfg.logger and "wandb" in cfg.logger:
            cfg.logger.wandb.offline = True

        pretty_print(cfg)
    return cfg


def empty(*args, **kwargs):
    pass

def restore(cfg: DictConfig) -> DictConfig:
    api = wandb.Api()
    run_path = None
    for run in api.runs(f"cnut1648/{cfg.logger.wandb.project}"):
        if str(cfg.test_from_run) in run.name:
            # run.path = [entity, project, id]
            run_path = os.path.join(*run.path)
            break
    assert run_path is not None, f"{cfg.test_from_num} is not in the wandb"
    orig_override: DictConfig = OmegaConf.load(
        wandb.restore("hydra/overrides.yaml", run_path=run_path))
    current_overrides = HydraConfig.get().overrides.task
    # concatenating the original overrides with the current overrides
    overrides: DictConfig = orig_override + current_overrides

    # getting the config name from the previous job.
    hydra_config = OmegaConf.load(
        wandb.restore("hydra/hydra.yaml", run_path=run_path))
    config_name: str = hydra_config.hydra.job.config_name

    cfg = compose(config_name, overrides=overrides)

    return cfg

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """
    This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """
    print(trainer.global_rank, trainer.is_global_zero)

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["module"] = config["module"]
    hparams["data"] = config["data"]
    hparams['seed'] = config['seed']
    hparams['debug'] = config['debug']
    hparams['data_dir'] = config['data_dir']
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "logger" in config:
        hparams["logger"] = config["logger"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(loggers):
    # make sure wandb is done
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb.finish()
