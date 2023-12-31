import torch
import lightning.pytorch as pl
# from timm.utils import ModelEmaV2
# from hydra.utils import instantiate


class kNNPredictor:
    def __init__(self, k=32, alpha=0.7):
        self.k = k

    def __call__(self):
        return


class NetworkModule(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        metric_fn,
        predictor=kNNPredictor()
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.predictor = predictor

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xs, y = batch
        x = torch.cat(xs)
        y_hat, emb = self.model(x)
        loss = self.loss_fn(emb, y_hat, y)
        score = self.metric_fn(y_hat, y)

        self.log("train_loss", loss["loss"], sync_dist=True)
        self.log("train_scl", loss["scl"], sync_dist=True)
        self.log("train_ce", loss["ce"], sync_dist=True)
        self.log("train_score", score, sync_dist=True)
        return loss["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        xs, y = batch
        x = torch.cat(xs)
        y_hat, emb = self.model(x)
        loss = self.loss_fn(emb, y_hat, y)
        score = self.metric_fn(y_hat, y)

        self.log("val_loss", loss["loss"], sync_dist=True)
        self.log("val_scl", loss["scl"], sync_dist=True)
        self.log("val_ce", loss["ce"], sync_dist=True)
        self.log("val_score", score, sync_dist=True)
        return loss["loss"]

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        score = self.metric_fn(y_hat, y)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_score", score, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            weight_decay=0.0005,
            momentum=0.9,
            nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            eta_min=0.,
            T_max=200
        )
        return {"optimizer": optim, "lr_scheduler": scheduler}

    # def on_before_zero_grad(self, *args, **kwargs):
    #     self.ema.update(self.model)


# ip=
