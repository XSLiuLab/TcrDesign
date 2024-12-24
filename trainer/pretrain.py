import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction (optional)

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.001, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, has_next=False, log_dir=None, 
                 has_warmup=True, LmWeight=None, batch_num=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        :param batch_num: how many batch to use, [None] uses all
        """
        self.batch_num = batch_num
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.has_next = has_next
        self.has_warmup = has_warmup
        self.LmWeight = LmWeight
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(self.bert, vocab_size, has_next).to(self.device)
        if LmWeight is not None:
            print("Loading pretained weight...")
            self.model.load_state_dict(torch.load(LmWeight, map_location=self.device))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps, limit_lr=lr, has_warmup=has_warmup)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

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
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
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
        total_correct = 0
        total_element = 0
        flag = (self.batch_num is not None) and train

        for i, data in data_iter:
            if flag and i >= self.batch_num:
                break
                    
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            if self.has_next:
                next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            else:
                mask_lm_output = self.model.forward(data["bert_input"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            if self.has_next:
                next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            if self.has_next:
                loss = next_loss + mask_loss
            else:
                loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            if self.has_next:
                correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                total_correct += correct
                total_element += data["is_next"].nelement()
            
            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100 if self.has_next else 0,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            
            # 保存关键信息 - 按step保存
            if self.log_dir is not None:
                self.writer.add_scalar(f'pretrain/{str_code}_loss_step', loss.item(), 
                                       epoch*self.batch_num+i if flag else epoch*len(data_iter)+i)
                self.writer.add_scalar(f'pretrain/{str_code}_lr_step', 
                                       self.optim_schedule._optimizer.param_groups[0]['lr'], 
                                       epoch*self.batch_num+i if flag else epoch*len(data_iter)+i)

        # 保存关键信息 - 按epoch保存
        if self.log_dir is not None:
            self.writer.add_scalar(f'pretrain/{str_code}_loss_epoch', 
                                   avg_loss / self.batch_num if flag else avg_loss / len(data_iter), epoch)
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), 
              avg_loss / self.batch_num if flag else avg_loss / len(data_iter), 
              "total_acc=", total_correct / total_element * 100 if self.has_next else 0)   

        return avg_loss / self.batch_num if flag else avg_loss / len(data_iter)


    def save(self, epoch, file_path: str):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
