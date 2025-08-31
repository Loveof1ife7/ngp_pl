import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scripts.img.opt import get_opts

# data
from torch.utils.data import DataLoader
from datasets.imager import ImageDataset
from scripts.img.utils import write_image
from scripts.img.common import read_image

# models
from models.image.networks import ImageSiren

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import StepLR

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils import slim_ckpt, load_ckpt

import time
import commentjson as json
from utils import load_ckpt, seed_everything, process_batch_in_chunks

import warnings; warnings.filterwarnings("ignore")

class SirenImageSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        
        ### Create the experiment directory
        exp_dir = os.path.join(self.hparams.output_dir, self.time)
        
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            
        ### Load the configuration file
        with open(self.hparams.config) as config_file:
            self.config = json.load(config_file)
        
        config_path = f"{exp_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, separators=(',', ': '), sort_keys=True)
            
        ### save validation results
        self.val_metric_log_path = os.path.join(exp_dir, "val_metrics.txt") 
        with open(self.val_metric_log_path, 'w') as f:
            f.write("epoch\tpsnr\tssim\n")
        
        ### load image data
        self.img_data = torch.from_numpy(read_image(self.hparams.input_path)).float()
        print(f"[DEBUG] loaded image shape: {self.img_data.shape}")

        ### psnr and ssim
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    

    def setup(self, stage):
        ### model
        self.model = ImageSiren(
            img_channels=3,
            num_layers=self.config["network"]["num_layers"],
            hidden_dim=self.config["network"]["hidden_dim"],
        )
        
        ema_decay = 0.95
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            ema_decay * averaged_model_parameter + (1-ema_decay) * model_parameter
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # dataset
        self.train_dataset = ImageDataset(data=self.img_data,
                                          size=1000,
                                          num_samples=self.hparams.batch_size,
                                          split='train')

        self.test_dataset = ImageDataset(data=self.img_data,
                                         size=1,
                                         num_samples=self.hparams.batch_size,
                                         split='test')

    def forward(self,batch):
        b_pos = batch['points']
        pred = self.model(b_pos)
        return pred

    def on_fit_start(self):
        seed_everything(self.hparams.seed)
        
    def configure_optimizers(self):
        load_ckpt(self.model, self.hparams.ckpt_path)

        opts = []
        net_params = self.model.get_params(self.config["training"]["LR_scheduler"])
        self.net_opt = FusedAdam(net_params, betas=(0.9, 0.99), eps=1e-15)
        opts += [self.net_opt]

        lr_interval = self.config["training"]["LR_scheduler"][0]["interval"]
        lr_factor = self.config["training"]["LR_scheduler"][0]["factor"]

        if self.config["training"]["LR_scheduler"][0]["type"] == "Step":
            net_sch = StepLR(self.net_opt, step_size=lr_interval, gamma=lr_factor)
        else:
            net_sch = None

        return opts, [net_sch]


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        num_workers=16,
                        persistent_workers=True,
                        batch_size=None,
                        pin_memory=True)


    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                        num_workers=8,
                        batch_size=None,
                        pin_memory=True)


    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                        num_workers=8,
                        batch_size=None,
                        pin_memory=True)
    def training_step(self, batch, batch_nb, *args):
        results = self(batch)

        b_occ = batch['rgbs'].to(results.dtype)
        loss = F.mse_loss(results, b_occ)
        #batch_loss = (results - b_occ)**2 / (b_occ.detach()**2 + 1e-2)
        #loss = batch_loss.mean()

        self.log('lr/network', self.net_opt.param_groups[0]['lr'], True)
        self.log('train/loss', loss)

        return loss


    def training_epoch_end(self, training_step_outputs):
        for name, cur_para in self.model.named_parameters():
            if len(cur_para) == 0:
                print(f"The len of parameter {name} is 0 at epoch {self.current_epoch}.")
                continue

            if cur_para is not None and cur_para.requires_grad and cur_para.grad is not None:
                para_norm = torch.norm(cur_para.grad.detach(), 2)
                self.log('Grad/%s_norm' % name, para_norm)


    def on_before_zero_grad(self, optimizer):
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)


    def backward(self, loss, optimizer, optimizer_idx):
        # do a custom way of backward to retain graph
        loss.backward(retain_graph=True)


    def on_train_start(self):
        gt_img = self.img_data.reshape(self.img_data.shape).float().clamp(0.0, 1.0)
        gt_img = gt_img.cpu().numpy()

        img_path = f'{self.hparams.output_dir}/{self.time}/reference.jpg'
        
        if self.img_data.shape[-1] == 1:
            self.img_data = self.img_data.repeat(1, 1, 3)

        write_image(img_path, gt_img)
        print(f"\nWriting '{img_path}'... ", end="")


        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log("misc/model_size", model_size)
        print(f"\nThe model size: {model_size}")


    def on_train_end(self):
        # The final validation will use the ema model, as it replaces our normal model
        if self.ema_model is not None:
            print("Replacing the standard model with the EMA model for last validation run")
            self.model = self.ema_model


    def on_validation_start(self):
        torch.cuda.empty_cache()

        if not self.hparams.no_save_test:
            self.val_dir = f'{self.hparams.output_dir}/{self.time}/validation/'
            os.makedirs(self.val_dir, exist_ok=True)


    def validation_step(self, batch, batch_nb):
        img_size = self.img_data.shape[0] * self.img_data.shape[1]

        pred_img = process_batch_in_chunks(batch["points"], self.ema_model, max_chunk_size=2**18)
        pred_img = pred_img[:img_size, :].reshape(self.img_data.shape).float().clamp(0.0, 1.0)

        gt_img = self.img_data.float().clamp(0.0, 1.0).to(pred_img.device)

        # tensor-based SSIM expects (N, C, H, W)
        gt = gt_img.permute(2, 0, 1).unsqueeze(0)     # [1, C, H, W]
        pred = pred_img.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]    

        psnr = self.val_psnr(gt, pred)
        ssim = self.val_ssim(gt, pred)

        if not self.hparams.no_save_test:
            img_path = f"{self.val_dir}/{self.current_epoch}.jpg"
            write_image(img_path, pred_img)

        self.log("val/psnr", psnr, prog_bar=True)
        self.log("val/ssim", ssim, prog_bar=True)

        return {"psnr": psnr, "ssim": ssim}


    def validation_epoch_end(self, outputs):
        psnrs = [x['psnr'] for x in outputs]
        ssims = [x['ssim'] for x in outputs]

        avg_psnr = sum(psnrs) / len(psnrs)
        avg_ssim = sum(ssims) / len(ssims)
        
        with open(self.val_metric_log_path, 'a') as f:
            f.write(f"{self.current_epoch}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")

        # 同时记录到 tensorboard
        self.log("val/avg_psnr", avg_psnr)
        self.log("val/avg_ssim", avg_ssim)

        with open(self.val_metric_log_path, 'a') as f:
            f.write(f"{self.current_epoch}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")

    def predict_step(self, batch, batch_idx):
  
        H, W, C = self.img_data.shape
        device = self.device
        
        gt_img = self.img_data.float().clamp(0.0, 1.0).to(device)  # [H, W, C]
        if C == 1:
            gt_img = gt_img.repeat(1, 1, 3)
            C = 3 
            
        print(f"[DEBUG] Device: {device}, Image shape: {H}x{W}x{C}")

        # 构造 [-1,1] 范围的坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        coords = torch.stack((x_coords, y_coords), dim=-1).reshape(-1, 2)  # [H*W, 2]
        
        print(f"[DEBUG] x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}")
        print(f"[DEBUG] coords shape: {coords.shape}")

        pred_img = process_batch_in_chunks(coords, self.ema_model, max_chunk_size=2**18)  # [H*W, C]
        pred_img = pred_img[:H*W, :].reshape(H, W, 3).float().clamp(0.0, 1.0).to(device)  # [H, W, C]
        
        # 转为 [1, C, H, W] 计算 PSNR / SSIM
        pred = pred_img.permute(2, 0, 1).unsqueeze(0)
        gt = gt_img.permute(2, 0, 1).unsqueeze(0)
        
        psnr = self.val_psnr(gt, pred)
        ssim = self.val_ssim(gt, pred)

        # 保存图像
        save_dir = f"{self.hparams.output_dir}/{self.time}/predict"
        os.makedirs(save_dir, exist_ok=True)
        write_image(f"{save_dir}/result.jpg", pred.squeeze(0))  # shape [C, H, W]

        print(f"[PREDICT] PSNR: {psnr.item():.4f}, SSIM: {ssim.item():.4f}")
        return {"psnr": psnr.item(), "ssim": ssim.item()}


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def lr_scheduler_step(self, scheduler, metric, optimizer_idx=None):
        scheduler.step(metric)
        
if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = SirenImageSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'{hparams.output_dir}/{system.time}/ckpts/',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"{hparams.output_dir}/{system.time}/logs/",
                               name="",
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=1,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      gradient_clip_val=1.0,
                      strategy=None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    if hparams.val_only:
        trainer.predict(system, ckpt_path=hparams.ckpt_path)
        system.output_metrics(logger)
    else:
        #trainer.fit(system, ckpt_path=hparams.ckpt_path)
        trainer.fit(system)
        trainer.predict()
        system.output_metrics(logger)