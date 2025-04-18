import os, json, pickle, torch, tqdm
torch.manual_seed(42)
from pathlib import Path
import wandb
wandb.login()
from transformers import TrainingArguments, Trainer, DefaultDataCollator, get_linear_schedule_with_warmup
from dataset import COCODatasetOnDemand
from models import VisionModelForCLS
import numpy as np
from typing import List, Tuple, Union, Any
if __name__ == "__main__": # Commented out for execution in interactive env
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class ModelTrainer:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.model_name = self.config["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]

        self.train_ds = COCODatasetOnDemand(root_dir=self.config["root_dir"], frac=self.config["frac"], is_train=True, privileged_indices_set=self.config["privileged_indices_set"])
        self.eval_ds = COCODatasetOnDemand(root_dir=self.config["root_dir"], frac=self.config["frac"], is_train=False, privileged_indices_set=self.config["privileged_indices_set"])
        self.data_collator = DefaultDataCollator(return_tensors="pt")

        self.batch_size = self.config["batch_size"]
        self.num_epochs = self.config["num_epochs"]
        self.num_steps = len(self.train_ds)//(self.batch_size)
        self.project_name = self.config["project_name"] + f"_{self.model_name_srt}"
        self.ft = self.config['finetune_module']
        self.wandb_log = self.config["wandb_log"]
        
        if self.ft == "ref_model":
            self.run_name = f"{self.ft}_{self.config['frac']:.2f}_{self.config['initial_lr']:.1e}"
            self.model = VisionModelForCLS(
                device=device,
                model_name=self.model_name,
                num_labels=len(self.train_ds.label_names),
                ref_cls_weights_path=None,
                privileged_indices=self.train_ds.privileged_indices,
                non_privileged_indices=self.train_ds.non_privileged_indices,
                beta=self.config["beta"],      # Example value
                epsilon=self.config["epsilon"],   # Example value
                is_ref=True,  # Set to True for reference model
                quant_config=None
            ).to(device)
        else:
            raise ValueError("Invalid finetune module!")
            
        self.checkpoint_dir = Path(f"output/ckpt/{self.project_name}/{self.run_name}")
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        self.eval_dl = torch.utils.data.DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.data_collator)

        if self.wandb_log:
            wandb.init(project=self.project_name, name=self.run_name, config=config)
            wandb.watch(self.model, log="all")
            wandb.define_metric("train/step")
            wandb.define_metric("val/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="val/step")
    
    def _find_norm(self, is_grad: bool) -> float:
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                if is_grad:
                    k = val
                else:
                    k = val.grad if val.grad is not None else torch.tensor(0.0, device=self.device)
                norm += (k ** 2).sum().item()
        norm = norm ** 0.5  
        return norm
    
    def _forward_batch(self, batch: dict, is_train: bool) -> dict:
        pixels = batch["pixels"].to(self.device) # (b, 3, 224, 224)
        labels = batch["labels"].to(self.device) # (b, c)
        if is_train:
            self.model.train()
            out = self.model(pixels, labels)
            out["outputs"].requires_grad_(True)
            assert out["outputs"].requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(pixels, labels)
        return out
    
    def _optimize_batch(self, batch: dict, ep: int, batch_index: int) -> Tuple[float]:  
        out = self._forward_batch(batch, is_train=True)
        loss = out["loss"]["loss"]
        acc = out["acc"]["acc"]
        loss.backward()     
        gn = self._find_norm(True) 
        pn = self._find_norm(False) 
        lr = self.optimizer.param_groups[0]['lr'] 
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config["max_grad_norm"], norm_type=2.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step(ep + batch_index/len(self.train_dl))
        return loss.item(), acc, gn, pn, lr

    def _optimize_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.train_dl, desc=f"[TRAIN] ep: {ep}/{self.num_epochs-1}", total=len(self.train_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                loss, acc, gn, pn, lr = self._optimize_batch(batch=batch, ep=ep, batch_index=i)
                if self.wandb_log:
                    wandb.log({"train/loss": loss, "train/accuracy": acc, "train/learning_rate": lr, "train/grad_norm": gn, "train/param_norm": pn, "train/epoch": ep, "train/step": self.train_step})
                    self.train_step += 1
                pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}", "lr": f"{lr:.3e}"})                        
    
    def _validate_dataloader(self, ep: int) -> None:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[VAL] ep: {ep}/{self.num_epochs-1}", total=len(self.eval_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                out = self._forward_batch(batch=batch, is_train=False) 
                loss = out["loss"]["loss"]
                acc = out["acc"]["acc"]
                if self.wandb_log:
                    wandb.log({"val/loss": loss, "val/accuracy": acc, "val/epoch": ep, "val/step": self.val_step})
                    self.val_step += 1
                pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"})  
    
    def _save_checkpoint(self, ep: int, is_latest_ckpt: bool) -> None:
        checkpoint = {"epoch": ep, 
                      "model_state": self.model.state_dict(), 
                      "opt_state": self.optimizer.state_dict(),
                      "config": self.config}   
        if not Path.exists(self.checkpoint_dir):
            Path.mkdir(self.checkpoint_dir, parents=True, exist_ok=True)
        if is_latest_ckpt:
            checkpoint_path = Path(self.checkpoint_dir, f"ckpt_ep_latest.pth")
        else:
            checkpoint_path = Path(self.checkpoint_dir, f"ckpt_ep_{ep}.pth")            
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] ep: {ep}/{self.num_epochs-1}, checkpoint saved at: {checkpoint_path}")
    
    def _load_checkpoint(self, ckpt_file_path: str) -> None:
        checkpoint_path = Path(ckpt_file_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")
    
    def train(self) -> None:
        self.model.calc_num_params()
        self.train_step = 0
        self.val_step = 0
        for ep in range(self.num_epochs):
            self._optimize_dataloader(ep=ep)
            self._validate_dataloader(ep=ep)
            self._save_checkpoint(ep=ep, is_latest_ckpt=True)
        if self.wandb_log:
            wandb.finish()
    
def main(device: torch.device) -> None:
    config = {
        "model_name": "google/vit-base-patch16-224",
        "finetune_module": "ref_model",
        "root_dir": "/raid/speech/soumen/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014",
        "project_name": "fairpo_ref_model_finetune",
        "ref_cls_weights_path": None,
        "privileged_indices_set": None, # Set of privileged indices
        "beta": 1.0, # DPO beta
        "epsilon": 0.1, # non privileged loss margin
        "num_epochs": 10,
        "batch_size": 16,
        "frac": 1.0,
        "initial_lr": 5e-5,
        "max_grad_norm": 10.0,
        "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999),
        "num_ckpt_per_epoch": 2,
        "wandb_log": True
    }

    trainer = ModelTrainer(device=device, config=config)
    trainer.train()
    # trainer._load_checkpoint(ckpt_file_path="output/ckpt/fairpo_ref_model_finetune_vit-base-patch16-224/ref_model_0.00_5.0e-05/ckpt_ep_latest.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    main(device=device)