from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.trainer import Trainer
import lightning.pytorch as pl
import hydra
from hydra.utils import instantiate

# from nats_bench import create
# from xautodl.models import get_cell_based_tiny_net

@hydra.main(version_base=None, config_path="zebanas/configs/classification", config_name="train")
def main(cfg):
    pl.seed_everything(777, workers=True)

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
    # api = create(cfg.api_path, 'tss', fast_mode=True, verbose=True)
    # config = api.get_net_config(23, 'cifar10')
    # model = get_cell_based_tiny_net(config)
    model = instantiate(cfg.module)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
