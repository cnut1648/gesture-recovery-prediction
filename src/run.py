from typing import List
import os
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule, Trainer, Callback, LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.utils.conf import log_hyperparameters, finish, restore
from src.utils.utils import (
    get_logger
)

log = get_logger(__name__)

# @rank_zero_only
def get_pl_logger(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
                while True:
                    try:
                        # sometimes fail for unknown reason
                        print(logger.experiment)
                        break
                    except BaseException:
                        pass

                if "wandb" in lg_conf["_target_"]:
                    if not cfg.debug:
                        # will upload this run to cloud in the end of the run
                        log.info(f"wandb url in {logger.experiment.url}")
                        # get id from x-y-id
                        id = logger.experiment.name.rsplit('-', 1)[1]
                        cfg.callback.model_checkpoint.dirpath = os.path.join(
                            cfg.callback.model_checkpoint.dirpath, id, "ckpt"
                        )
                    else:
                        # no wandb in this case
                        id = "offline"

    return loggers


def get_pl_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callback" in cfg:
        for cb_name, cb_conf in cfg["callback"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback {cb_name} <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

    
def run(cfg: DictConfig) -> float:
    """init data, model and call trainer"""
    # TODO
    # change to lightning
    
    seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        # ignore instantiate of aug
        _recursive_=False
    )
    log.info(f"data <{cfg.data._target_}> loaded")

    module: LightningModule = hydra.utils.instantiate(
        cfg.module, optcfg=cfg.module.optim, schcfg=getattr(cfg.module, "scheduler", None),
        # use_test=cfg.data.use_test, transforms=cfg.data.transforms,
        # non-recursive to enable optim / scheduler instantiate
        _recursive_=False
    )
    log.info(f"model {cfg.module.model.arch} <{cfg.module.model._target_}> loaded")

    loggers: List[LightningLoggerBase] = get_pl_logger(cfg)

    callbacks: List[Callback] = get_pl_callbacks(cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks, logger=loggers,
        _convert_="partial"
    )

    log.info(f"log info (hydra override or csv or main.log) in {os.getcwd()}")

    if cfg.test_from_run is not None:
        # api 
        # now the dirpath already modified in getting wandb logger
        # thus rm added part (this wandb run's id) and append id of ckpt
        ckpt_path = Path(cfg.output_dir) / str(cfg.test_from_run) / "ckpt"
        assert ckpt_path.exists()

        model_ckpt = None
        for ckpt in os.listdir(ckpt_path):
            if ckpt == "last.ckpt":
                continue
            model_ckpt = ckpt_path / ckpt
        module.load_from_checkpoint(model_ckpt)
        trainer.test(model=module, datamodule=datamodule)
        return 0.0



    log_hyperparameters(
        config=cfg,
        model=module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )
    log.info("hyperparams saved")

    log.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    log.info("start training")
    trainer.fit(model=module, datamodule=datamodule)
    if cfg.data.has_test:
        log.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.info("testing")
        trainer.test()

    ckpt = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{ckpt}")

    # return for optuna tune
    if "Reg" in cfg.module._target_:
        # mse 
        metric = trainer.callback_metrics["valid/epoch/loss"]
        log.info(f"metric to tune MSE: {metric}")
    else:
        import re
        metric = float(re.split(r'AUROC([\d.]+)', ckpt)[1])
        log.info(f"metric to tune AUROC: {metric}")
    # metric = None
    # if "tune_metric" in cfg:
    #     if type(cfg.tune_metric) is list:
    #         metric = tuple([
    #           trainer.callback_metrics[m].item()
    #             for m in cfg.tune_metric
    #         ])
    #     else:
    #         metric = trainer.callback_metrics[cfg.tune_metric].item()
    #     log.info(f"metric to tune: {cfg.tune_metric}: {metric}")

    finish(loggers)
    return metric
