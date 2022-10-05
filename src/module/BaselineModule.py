from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from src.utils.utils import get_logger
from omegaconf import DictConfig
from einops import rearrange
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import ModuleDict
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, MetricCollection, Recall, Precision, F1, PrecisionRecallCurve, AUC

log = get_logger(__file__)
class BaselineModule(pl.LightningModule):
    def __init__(
            self, 
            model: DictConfig, optcfg: DictConfig,
            # control data aug
            # transforms: DictConfig, label_smoothness: float,
            # use_test: bool = False, flow_method: str = "none",
            r_drop_weight: float = 0,
            schcfg: Optional[DictConfig] = None,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module = hydra.utils.instantiate(
            model, _recursive_=False
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # (float logits (pre-sigmoid), long label)
        metric = MetricCollection([
            Accuracy(threshold=0.0),
            Recall(threshold=0.0),
            Precision(threshold=0.0),
            F1(threshold=0.0),
            # Accuracy(threshold=-0.0020),
            # Recall(threshold=-0.0020),
            # Precision(threshold=-0.0020),
            # F1(threshold=-0.0020),
            AUROC(pos_label=1)
        ])
        self.prc = PrecisionRecallCurve(pos_label=1)
        auc =  AUC(reorder=True)
        self.metrics: Dict[str, MetricCollection] = ModuleDict()
        for phase in ["train", "valid", "test"]:
            self.metrics[f"{phase}_metric"] = metric.clone()
            self.metrics[f"{phase}_auc"] = auc.clone()

    def r_drop_kl_loss(self, logits_p, logits_q):
        """
        kl for R-Drop
        https://arxiv.org/pdf/2106.14448.pdf

        Args:
            logits_p (N, )
            logits_q (N, )
        """
        # or reduction = sum
        p_kl: float = F.kl_div(logits_p.log_softmax(dim=0), logits_q.softmax(dim=0), 
            reduction="batchmean")
        q_kl: float = F.kl_div(logits_q.log_softmax(dim=0), logits_p.softmax(dim=0), 
            reduction="batchmean")
        return (p_kl + q_kl) / 2

    ########
    # agg after step & epoch
    ########

    def compute_metric_and_log(self, output: dict, phase: str):
        """
        must compute metric and log in <phase>_step_end
        output: dict returned by forward
        """
        loss, logits, label = output["loss"], output["logits"], output["label"].long()

        metrics: MetricCollection = self.metrics[f"{phase}_metric"]
        self.log(f"{phase}/step/loss", loss)

        metrics.update(logits, label)
        precision, recall, _ = self.prc(logits, label)
        self.metrics[f'{phase}_auc'].update(recall, precision)
        return

        # compute metric
        try:
            m = metrics(logits, label)
            for metric_name, metric in metrics.items():
                self.log(f"{phase}/step/{metric_name}", metric, logger=True,
                            # on_step if train; on_epoch if not train
                            on_step=phase == "train", on_epoch=phase != "train")
        except:
            pass

        self.log(f"{phase}/step/loss", loss)

    def agg_epoch(self, outputs: EPOCH_OUTPUT, phase: str):
        loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.log(f"{phase}/epoch/loss", loss)

        # log aggregate metric
        metrics: MetricCollection = self.metrics[f"{phase}_metric"]
        for metric_name, metric in metrics.items():
            self.log(f"{phase}/epoch/{metric_name}", metric.compute(), logger=True, prog_bar=(phase != "train"),
                        # on_step if train; on_epoch if not train
                        on_step=False, on_epoch=True)
            metric.reset()
        auc = self.metrics[f"{phase}_auc"]
        self.log(f"{phase}/epoch/AUPRC", auc.compute(), logger=True, prog_bar=(phase != "train"), on_step=False, on_epoch=True)
        auc.reset()

    ########
    # one step
    ########

    def forward(self, batch: dict):
        """
        inference, return logits
        batch key
        - L
        - R
        - label (N, ) long
        - scores: only when in test

        return (N, ) logits
        """
        logits = self.model(**batch)
        return logits

    def step(self, batch: dict):
        """
        one step, return logits, and loss
        pred = logits > 0
        """
        # long tensor
        label = batch['label']
        # TODO
        # check label smoothing here
        # float tensor so that can be computed by criterion (accept float)
        smoothed_label = (label - self.hparams.label_smoothness).abs()
        logits = self(batch)
        loss = self.criterion(logits, smoothed_label)
        return logits, loss

    ########
    # train
    ########

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        # video augmentation
        # batch["video"] = self.transforms(batch["video"], is_train=True)
        logits, loss = self.step(batch)
        if self.hparams.r_drop_weight > 0:
            # since dropout, not same as logits & loss
            # min KL(loss || loss2)
            logits2, loss2 = self.step(batch)
            loss = (loss + loss2) / 2 + self.hparams.r_drop_weight * self.r_drop_kl_loss(logits, logits2)
            logits = (logits + logits2) / 2
        return {"loss": loss, "logits": logits.detach(), "label": batch['label']}

    def training_step_end(self, output: dict) -> dict:
        self.compute_metric_and_log(output, "train")
        return output

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "train")

    ########
    # validate
    ########

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        # batch["video"] = self.transforms(batch["video"], is_train=False)
        logits, loss = self.step(batch)

        return {"loss": loss, "logits": logits.detach(), "label": batch['label']}

    def validation_step_end(self, output: dict) -> dict:
        self.compute_metric_and_log(output, "valid")
        return output

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "valid")

    ########
    # test
    ########

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        # batch["video"] = self.transforms(batch["video"], is_train=False)
        logits, loss = self.step(batch)

        return {"loss": loss, "logits": logits.detach(), "label": batch['label']}
    
    def test_step_end(self, output: dict) -> dict:
        self.compute_metric_and_log(output, "test")
        return output

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "test")

    # def test_step(self, batch: dict, batch_idx: int) -> dict:
    #     # batch["video"] = self.transforms(batch["video"], is_train=False)
    #     logits, loss = self.step(batch)

    #     # simple baseline based on skill alone
    #     # heuristic see EDA.ipynb
    #     # binary decision directly as prob (i.e. super confident)
    #     # scores: dict = batch["scores"]
    #     # naive_baseline = torch.logical_and(scores["ave_th"] < 2.7, scores["ave_tr"] > 2.7)
    #     # from sklearn.metrics import roc_auc_score
    #     # self.log("test/epoch/naive_baseline", roc_auc_score(batch["label"].detach().cpu().numpy(), naive_baseline.detach().cpu().numpy()))

    #     # simple baseline based on nerve sparing
    #     # binary decision directly as prob (i.e. super confident)
    #     from sklearn.metrics import roc_auc_score
    #     # (N, )
    #     ns = batch["L"]["nerve sparing"] + batch["R"]["nerve sparing"]
    #     ns_baseline1 = (ns == 4)
    #     self.log("test/epoch/ns_baseline1", 
    #             roc_auc_score(batch["label"].detach().cpu().numpy(), ns_baseline1.detach().cpu().numpy()))

    #     def baseline2(num):
    #         if num == 4: return 1
    #         if num == 1: return 0
    #         if num == 2: return torch.bernoulli(torch.tensor(0.25))
    #         if num == 3: return torch.bernoulli(torch.tensor(0.75))
    #         raise NotImplementedError("shouldn't has 0 or >4 ns")

    #     ns_baseline2 = torch.tensor(list(map(baseline2, ns)))
    #     self.log("test/epoch/ns_baseline2",
    #             roc_auc_score(batch["label"].detach().cpu().numpy(), ns_baseline2.detach().cpu().numpy()))

    #     return {"loss": loss, "logits": logits, "label": batch['label']}

    # def test_step_end(self, output: dict) -> dict:
    #     self.compute_metric_and_log(output, "test")
    #     return output

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     self.agg_epoch(outputs, "test")

    ########
    # optim
    ########

    def configure_optimizers(self):
        # set BN weight_decay = 0
        bn_params = []
        non_bn_params = []
        for name, p in self.named_parameters():
            if "bn" in name:
                bn_params.append(p)
            else:
                non_bn_params.append(p)

        optim_params = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": non_bn_params, "weight_decay": self.hparams.optcfg.weight_decay},
        ]
        opt = hydra.utils.instantiate(
            self.hparams.optcfg, params=optim_params,
            _convert_="all"
        )

        ret_dict = {'optimizer': opt}
        if self.hparams.schcfg:
            scheduler = hydra.utils.instantiate(
                self.hparams.schcfg, optimizer=opt,
                _convert_="all"
            )
            ret_dict["lr_scheduler"] = scheduler
            ret_dict["monitor"] = "valid/epoch/loss"

        return ret_dict
