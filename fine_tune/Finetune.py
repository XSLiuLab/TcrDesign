import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..trainer.optim_schedule import ScheduledOptim
from .tcr_model import BertForSequenceClassification
from .tcr_dataset import FineTuneDataset

from transformers import get_linear_schedule_with_warmup

import os
import tqdm

import numpy as np
from sklearn import metrics


class FinetuneTrainer:
    """
    FinetuneTrainer make the fine tune the pretrained BERT model to binding task.
    """
    
    def __init__(self, model: BertForSequenceClassification, 
                 train_dir: str, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_dir=None, 
                 has_warmup=True, warmup_steps=1000, **kwargs):
        
        # Setup cuda device for BERT fine-tuning, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.has_warmup = has_warmup
        self.warmup_steps = warmup_steps
        
        self.kwargs = kwargs
        self.train_dir = train_dir
        self.train_data_list = os.listdir(train_dir) * 20
        self.test_data = test_dataloader
        
        self.model = model.to(self.device)
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.alphaEncoder.hidden, n_warmup_steps=warmup_steps, limit_lr=lr, has_warmup=has_warmup)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(self.device)) # 正负样本不平衡
        # self.criterion = nn.BCEWithLogitsLoss()
        
        self.log_freq = log_freq
        
        self.log_dir = log_dir
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        train_dataset_path = os.path.join(self.train_dir, self.train_data_list[epoch])
        print("Loading Train Dataset", train_dataset_path)
        train_dataset = FineTuneDataset(data_path=train_dataset_path, **self.kwargs)
        train_data_loader = DataLoader(train_dataset, batch_size=self.test_data.batch_size, num_workers=self.test_data.num_workers, shuffle=True)
        loss = self.iteration(epoch, train_data_loader)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss
    
    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"
        
        # 调整模型模式
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        true_label = []
        pred_score = []

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            model_output = self.model.forward(data["aCDR3"], data["bCDR3"], data["pMHC"], data["VJ"], data["pMHC_segment_label"])
            # 计算损失
            loss = self.criterion(model_output.squeeze(-1), data["binding_score"])

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # binding prediction accuracy
            avg_loss += loss.item()
            true_label.extend(data["binding_score"].cpu().tolist())
            pred_score.extend(torch.sigmoid(model_output.squeeze(-1)).cpu().detach().tolist())

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            
            # 保存关键信息 - 按step保存
            if self.log_dir is not None:
                self.writer.add_scalar(f'finetune/{str_code}_loss_step', loss.item(), epoch*len(data_iter)+i)
                self.writer.add_scalar(f'finetune/{str_code}_lr_step', 
                                       self.optim_schedule._optimizer.param_groups[0]['lr'], epoch*len(data_iter)+i)
        
        # 保存关键信息 - 按epoch保存
        if self.log_dir is not None:
            true_label_array = np.array(true_label)
            pred_score_array = np.array(pred_score)
            fpr, tpr, _ = metrics.roc_curve(true_label_array, pred_score_array)
            precision, recall, _ = metrics.precision_recall_curve(true_label_array, pred_score_array)
            self.writer.add_scalar(f'finetune/{str_code}_roauc_epoch', round(metrics.auc(fpr, tpr), 4), epoch)
            self.writer.add_scalar(f'finetune/{str_code}_prauc_epoch', round(metrics.auc(recall, precision), 4), epoch)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        
        # return avg_loss / len(data_iter)
        return round(metrics.auc(recall, precision), 4)

    def save(self, epoch, file_path: str):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class Distance_FinetuneTrainer:
    """
    FinetuneTrainer make the fine tune the pretrained BERT model to distance prediction task.
    """
    
    def __init__(self, model: BertForSequenceClassification, 
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_dir=None, 
                 has_warmup=True, warmup_steps=1000, **kwargs):
        
        # Setup cuda device for BERT fine-tuning, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.has_warmup = has_warmup
        self.warmup_steps = warmup_steps
        
        self.kwargs = kwargs
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        self.model = model.to(self.device)
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.model.alphaEncoder.hidden, n_warmup_steps=warmup_steps, limit_lr=lr, has_warmup=has_warmup)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.model.alphaEncoder.hidden, n_warmup_steps=warmup_steps, limit_lr=lr, has_warmup=has_warmup)
        
        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()
        
        self.log_freq = log_freq
        
        self.log_dir = log_dir
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss
    
    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"
        
        # 调整模型模式
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            bs = data["aCDR3"].size(0)
            _ = self.model.forward(data["aCDR3"], data["bCDR3"], data["pMHC"], data["VJ"], data["pMHC_segment_label"])
            # 计算损失
            attn_weight = self.model.attention.attention.attn_weight # 收集attention权重
            attn_weight = attn_weight.mean(1) # (bs, bCDR3, pep)
            attn_weight = attn_weight[:, 1:-1, 36:-1]
            loss = self.criterion(attn_weight[:, :, :13].reshape(bs, -1), 
                                 (data["norm_distance_matrix"][:, :, :13]>=0.9).reshape(bs, -1).to(torch.float32)) # 专注于I型肽, 大于0.9认为是结合残基

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # binding prediction accuracy
            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            
            # 保存关键信息 - 按step保存
            if self.log_dir is not None:
                self.writer.add_scalar(f'finetune/{str_code}_loss_step', loss.item(), epoch*len(data_iter)+i)
                self.writer.add_scalar(f'finetune/{str_code}_lr_step', 
                                       self.optim_schedule._optimizer.param_groups[0]['lr'], epoch*len(data_iter)+i)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        
        return avg_loss / len(data_iter)

    def save(self, epoch, file_path: str):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
