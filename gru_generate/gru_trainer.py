import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..trainer.optim_schedule import ScheduledOptim
from .gru_model import AttenSeq2Seq
import tqdm
import numpy as np


class RNNTrainer(object):
    def __init__(self, rnn: AttenSeq2Seq, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_dir=None,
                 has_warmup=True, warmup_steps=1000):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.has_warmup = has_warmup
        self.warmup_steps = warmup_steps
        
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        self.model = rnn.to(self.device)
        
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for RNN" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.encoder.peptideEncoder.hidden, n_warmup_steps=warmup_steps, limit_lr=lr, has_warmup=has_warmup)
        self.criterion = nn.NLLLoss(ignore_index=0) # 即忽略padding的loss计算

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
        true_labels = []
        pred_labels = []
        
        for i, data in data_iter:
            data = {"rnn_encoder_input": data[0].to(self.device), "rnn_decoder_input": data[1].to(self.device),
                    "rnn_decoder_target": data[2].to(self.device)}

            rnn_output = self.model(data["rnn_encoder_input"], data["rnn_decoder_input"])
            loss = self.criterion(rnn_output.transpose(1, 2), data["rnn_decoder_target"])

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optim_schedule.step_and_update_lr()
            
            avg_loss += loss.item()
            
            # 计算预测准确率
            pred_array = torch.argmax(rnn_output, dim=-1).reshape(-1).detach().cpu().numpy()
            true_array = data["rnn_decoder_target"].reshape(-1).detach().cpu().numpy()
            ignore_section = true_array == 0
            true_labels.extend(list(true_array[~ignore_section]))
            pred_labels.extend(list(pred_array[~ignore_section]))

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item()
                }
                data_iter.write(str(post_fix))
            
            # 保存关键信息
            if self.log_dir is not None:
                self.writer.add_scalar(f'rnn/{str_code}_loss_step', loss.item(), epoch*len(data_iter)+i)
                self.writer.add_scalar(f'rnn/{str_code}_lr_step', 
                                       self.optim_schedule._optimizer.param_groups[0]['lr'], epoch*len(data_iter)+i)
        
        # 保存关键信息 - 按epoch保存
        if self.log_dir is not None:
            true_label_array = np.array(true_labels)
            pred_score_array = np.array(pred_labels)
            pred_acc = float(np.sum(true_label_array==pred_score_array) / len(true_label_array))
            self.writer.add_scalar(f'rnn/{str_code}_predAcc_epoch', round(pred_acc, 4), epoch)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        
        return avg_loss / len(data_iter)
    
    def save(self, epoch, file_path: str):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
