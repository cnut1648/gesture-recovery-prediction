import os

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from copy import deepcopy

TASK_NAME = {
    "AB": "AB_label_needle_positionB",
    "BC": "BC_label_needle_entry_angleC",
    "CE": "CE_label_needle_driving_1D",
    "FG": "FG_label_needle_driving_2FG"
}
# customize resolver to support ${multi:5,8}
OmegaConf.register_new_resolver(
    "multi", lambda ways, shots: ways * shots
)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # put inside to enable tab-completion
    from src.utils.conf import touch
    from src.run import run

    # prevent restored cfg override
    use_k_fold = cfg.data.use_k_fold
    # additional debug, check if all exist....
    cfg = touch(cfg)
    # now cfg is good

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if use_k_fold:
        split_by_surgeon = getattr(cfg.data, "split_by_surgeon", False)
        if cfg.data.has_test and not split_by_surgeon:
            k_range = 4
        elif split_by_surgeon:
            k_range = 12
        elif "MC" in cfg.data["_target_"]:
            # k_range = 100
            k_range = 50
        else:
            k_range = 4

        metrics = []
        for k in range(k_range):
            cfg_clone = deepcopy(cfg)
            with open_dict(cfg_clone):
                cfg_clone.data.k_fold = k
            metric = run(cfg_clone)
            print(f"{k} -> {metric}")
            metrics.append(metric)
        # avg
        metric = sum(m for m in metrics) / len(metrics)
        print(f"metric = {metric}")
    else:
        metric = run(cfg)
    return metric


if __name__ == "__main__":
    main()
