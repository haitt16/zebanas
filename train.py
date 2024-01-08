from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.trainer import Trainer

import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="zebanas/configs/classification", config_name="train")
def main(cfg):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        **cfg.callback.checkpoint
    )
    wandb_logger = WandbLogger(project=cfg.logger_name)

    trainer = Trainer(
        **cfg.trainer,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger
    )

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.module)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
