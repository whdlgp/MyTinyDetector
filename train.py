from pathlib import Path
import time
import yaml
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from block.model import Detector
from dataloader.pascalvoc_trainval import get_dataloader, VOC_CLASSES
from loss.loss import build_target, compute_loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = len(VOC_CLASSES)

        self.model = Detector(num_classes=self.num_classes).to(self.device)
        self.grid_size = config["img_size"] // self.model.stride
        self.optimizer = AdamW([
            {"params": self.model.backbone.parameters(), "lr": config["lr"] * 0.1},
            {"params": self.model.head.parameters(), "lr": config["lr"]},
        ], weight_decay=config["weight_decay"])
        # Warm up + CosineAnnealing LR scheduler
        warmup_epochs = config["warmup_epochs"]
        warmup_scheduler = LinearLR(self.optimizer, start_factor=config["warmup_start_factor"], total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=config["num_epochs"] - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        self.train_loader = get_dataloader(
            root=config["data_root"], image_set="train",
            img_size=config["img_size"], batch_size=config["batch_size"], shuffle=True,
        )
        self.val_loader = get_dataloader(
            root=config["data_root"], image_set="val",
            img_size=config["img_size"], batch_size=config["batch_size"], shuffle=False,
        )

        self.start_epoch = 0
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.get("resume"):
            self.load_checkpoint(config["resume"])

        self.log_path = Path(config["log_file"])
        if not self.log_path.exists():
            self.log_path.write_text(
                "epoch,"
                "train_loss,train_obj_loss,train_box_loss,train_cls_loss,"
                "train_tx_loss,train_ty_loss,train_w_loss,train_h_loss,"
                "val_loss,val_obj_loss,val_box_loss,val_cls_loss,"
                "val_tx_loss,val_ty_loss,val_w_loss,val_h_loss\n"
            )

    def save_checkpoint(self, epoch, filename):
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_obj_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0

        # Only for debugging
        total_tx_loss = 0.0
        total_ty_loss = 0.0
        total_w_loss = 0.0
        total_h_loss = 0.0

        # For small batch(GPU memory issue)
        # batch in 1 epoch: accum_steps x batch_size
        accum_steps = self.config["accum_steps"]
        self.optimizer.zero_grad()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for step, (images, boxes, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            output = self.model(images)
            target = build_target(boxes, labels, self.grid_size, self.num_classes).to(self.device)

            losses = compute_loss(output, target)
            loss = losses["total_loss"] / accum_steps

            loss.backward()
            if (step + 1) % accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += losses["total_loss"].item()
            total_obj_loss += losses["obj_loss"].item()
            total_box_loss += losses["box_loss"].item()
            total_cls_loss += losses["cls_loss"].item()

            # Only for debugging
            total_tx_loss += losses["tx_loss"].item()
            total_ty_loss += losses["ty_loss"].item()
            total_w_loss += losses["w_loss"].item()
            total_h_loss += losses["h_loss"].item()

            progress_bar.set_postfix(loss=f"{losses['total_loss'].item():.4f}")

        # Last optimizer step should be appled
        if len(self.train_loader) % accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_obj_loss = total_obj_loss / len(self.train_loader)
        avg_box_loss = total_box_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)

        # Only for debugging
        avg_tx_loss = total_tx_loss / len(self.train_loader)
        avg_ty_loss = total_ty_loss / len(self.train_loader)
        avg_w_loss = total_w_loss / len(self.train_loader)
        avg_h_loss = total_h_loss / len(self.train_loader)

        print(
            f"[epoch {epoch}] "
            f"train_loss {avg_loss:.4f} "
            f"obj_loss {avg_obj_loss:.4f} "
            f"box_loss {avg_box_loss:.4f} "
            f"cls_loss {avg_cls_loss:.4f}"
        )

        return {
            "total_loss": avg_loss,
            "obj_loss": avg_obj_loss,
            "box_loss": avg_box_loss,
            "cls_loss": avg_cls_loss,
            # Only for debugging
            "tx_loss": avg_tx_loss,
            "ty_loss": avg_ty_loss,
            "w_loss": avg_w_loss,
            "h_loss": avg_h_loss,
        }

    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_obj_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0

        # Only for debugging
        total_tx_loss = 0.0
        total_ty_loss = 0.0
        total_w_loss = 0.0
        total_h_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Validation]")
            for images, boxes, labels in progress_bar:
                images = images.to(self.device)
                output = self.model(images)
                target = build_target(boxes, labels, self.grid_size, self.num_classes).to(self.device)

                losses = compute_loss(output, target)
                loss = losses["total_loss"]

                total_loss += losses["total_loss"].item()
                total_obj_loss += losses["obj_loss"].item()
                total_box_loss += losses["box_loss"].item()
                total_cls_loss += losses["cls_loss"].item()
                
                # Only for debugging
                total_tx_loss += losses["tx_loss"].item()
                total_ty_loss += losses["ty_loss"].item()
                total_w_loss += losses["w_loss"].item()
                total_h_loss += losses["h_loss"].item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.val_loader)
        avg_obj_loss = total_obj_loss / len(self.val_loader)
        avg_box_loss = total_box_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)

        # Only for debugging
        avg_tx_loss = total_tx_loss / len(self.val_loader)
        avg_ty_loss = total_ty_loss / len(self.val_loader)
        avg_w_loss = total_w_loss / len(self.val_loader)
        avg_h_loss = total_h_loss / len(self.val_loader)

        print(
            f"[epoch {epoch}] "
            f"val_loss {avg_loss:.4f} "
            f"obj_loss {avg_obj_loss:.4f} "
            f"box_loss {avg_box_loss:.4f} "
            f"cls_loss {avg_cls_loss:.4f}"
        )

        return {
            "total_loss": avg_loss,
            "obj_loss": avg_obj_loss,
            "box_loss": avg_box_loss,
            "cls_loss": avg_cls_loss,
            # Only for debugging
            "tx_loss": avg_tx_loss,
            "ty_loss": avg_ty_loss,
            "w_loss": avg_w_loss,
            "h_loss": avg_h_loss,
        }

    def fit(self):
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            print(f"Epoch: {epoch}")
            epoch_start = time.time()

            if epoch < self.config["freeze_epochs"]:
                self.model.backbone.freeze()
            else:
                self.model.backbone.unfreeze()

            train_losses = self.train_one_epoch(epoch)
            val_losses = self.validate(epoch)

            # Print estimated remaining time
            elapsed = time.time() - epoch_start
            eta_hours = elapsed * (self.config["num_epochs"] - epoch - 1) / 3600
            print(f"[epoch {epoch}] {elapsed:.1f}s, ETA: {eta_hours:.2f}h")

            with open(self.log_path, "a") as f:
                f.write(
                    f"{epoch},"
                    f"{train_losses['total_loss']:.4f},"
                    f"{train_losses['obj_loss']:.4f},"
                    f"{train_losses['box_loss']:.4f},"
                    f"{train_losses['cls_loss']:.4f},"
                    # Only for debugging
                    f"{train_losses['tx_loss']:.4f},"
                    f"{train_losses['ty_loss']:.4f},"
                    f"{train_losses['w_loss']:.4f},"
                    f"{train_losses['h_loss']:.4f},"
                    
                    f"{val_losses['total_loss']:.4f},"
                    f"{val_losses['obj_loss']:.4f},"
                    f"{val_losses['box_loss']:.4f},"
                    f"{val_losses['cls_loss']:.4f},"
                    # Only for debugging
                    f"{val_losses['tx_loss']:.4f},"
                    f"{val_losses['ty_loss']:.4f},"
                    f"{val_losses['w_loss']:.4f},"
                    f"{val_losses['h_loss']:.4f}\n"
                )

            if (epoch + 1) % self.config["save_every"] == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch}.pth")

            self.save_checkpoint(epoch, "latest.pth")

            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                best_epoch = epoch
                self.save_checkpoint(epoch, "best.pth")

        print(f"Done, Best epoch: {best_epoch}")

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.fit()