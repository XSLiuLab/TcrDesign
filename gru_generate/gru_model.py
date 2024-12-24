import torch
import random
from torch import nn
from ..model import BERT
import torch.nn.functional as F


class SimpleRNNgenerate(nn.Module):
    def __init__(self, peptideEncoder: BERT, vocab_size,
                 output_size, kernel_size, pkernel_size, num_layer=1):
        super().__init__()
        
        self.peptideEncoder = peptideEncoder
        self.hidden_size = self.peptideEncoder.hidden
        self.output_size = output_size
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.pkernel_size = pkernel_size
        
        self.embed = nn.Embedding(vocab_size, self.hidden_size)
        self.drop = nn.Dropout(0.3)
        
        self.conv_block = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                                  out_channels=self.hidden_size,
                                                  kernel_size=self.kernel_size),
                                        nn.BatchNorm1d(self.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.MaxPool1d(kernel_size=(self.pkernel_size)))
        
        self.rnn_e = nn.GRU(self.hidden_size, self.hidden_size, self.num_layer, batch_first=True) # 第一维度为batch，而不是序列维度
        self.rnn_d = nn.GRU(self.hidden_size, self.hidden_size, self.num_layer, batch_first=True)
        
        self.out = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(self.hidden_size // 2, self.output_size))
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x_e, x_d):
        x_e = self.peptideEncoder(x_e)
        x_e = x_e.permute(0, 2, 1)
        x_e = self.conv_block(x_e)
        x_e = x_e.permute(0, 2, 1)
        _, hidden_e = self.rnn_e(x_e, None)
        
        x_d = self.drop(self.embed(x_d))
        x_d, _ = self.rnn_d(x_d, hidden_e)
        out = self.out(x_d)
        return self.softmax(out)

#########################################################
# 带注意力的Seq2Seq

class Encoder(nn.Module):
    def __init__(self, peptideEncoder: BERT, encoder_hidden_num, num_layer, decoder_hidden_num):
        super().__init__()
        self.peptideEncoder = peptideEncoder
        self.encoder_hidden_num = encoder_hidden_num
        self.num_layer = num_layer
        self.decoder_hidden_num = decoder_hidden_num
        
        self.rnn_e = nn.GRU(input_size = self.peptideEncoder.hidden,
                            hidden_size = encoder_hidden_num,
                            num_layers = num_layer,
                            bidirectional=True,
                            batch_first = True) # 双向，增强表示能力
        self.fc = nn.Linear(encoder_hidden_num * 2, decoder_hidden_num)
    
    def forward(self, x_e, init_hidden=None): # x_e [batch_size, seq_len]
        x_e = self.peptideEncoder(x_e) # x_e [batch_size, seq_len, hidden_num]
        encoder_output, last_hidden = self.rnn_e(x_e, init_hidden) # init_hidden=None 表示使用默认初始化
        # encoder_output [batch_size, seq_len, encoder_hidden_num * 2]
        # last_hidden [num_layer * 2, batch_size, encoder_hidden_num]
        h_m = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1) # 取最后一个时间步的隐藏输出，经过线性变换当作s0
        s0 = self.fc(h_m) # h_m [batch_size, encoder_hidden_num * 2]
        return encoder_output, s0 # s0 [batch_size, decoder_hidden_num]


class Encoder_both(nn.Module):
    def __init__(self, peptideEncoder: BERT, Bcdr3Encoder: BERT, 
                 encoder_hidden_num, num_layer, decoder_hidden_num, antigen_max_len=23):
        super().__init__()
        self.peptideEncoder = peptideEncoder
        self.Bcdr3Encoder = Bcdr3Encoder
        self.antigen_max_len = antigen_max_len
        self.encoder_hidden_num = encoder_hidden_num
        self.num_layer = num_layer
        self.decoder_hidden_num = decoder_hidden_num
        
        self.rnn_e = nn.GRU(input_size = self.peptideEncoder.hidden,
                            hidden_size = encoder_hidden_num,
                            num_layers = num_layer,
                            bidirectional=True,
                            batch_first = True) # 双向，增强表示能力
        self.fc = nn.Linear(encoder_hidden_num * 2, decoder_hidden_num)
    
    def forward(self, x_e, init_hidden=None): # x_e [batch_size, pep_seq_len + cdr3_seq_len]
        x_e_pep = x_e[:, :self.antigen_max_len]
        x_e_cdr3 = x_e[:, self.antigen_max_len:]
        x_e_pep = self.peptideEncoder(x_e_pep)
        x_e_cdr3 = self.Bcdr3Encoder(x_e_cdr3)
        x_e = torch.cat((x_e_pep, x_e_cdr3), dim=1)
        encoder_output, last_hidden = self.rnn_e(x_e, init_hidden)
        h_m = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1)
        s0 = self.fc(h_m)
        return encoder_output, s0


class Atten(nn.Module):
    def __init__(self, encoder_hidden_num, decoder_hidden_num):
        super().__init__()
        self.attnCal = nn.Linear((encoder_hidden_num * 2) + decoder_hidden_num, decoder_hidden_num, bias=False)
        self.v = nn.Linear(decoder_hidden_num, 1, bias=False) # bias设置为0，只进行矩阵运算
    
    def forward(self, s, encoder_output):
        # s [batch_size, decoder_hidden_num]
        # encoder_output [batch_size, seq_len, encoder_hidden_num * 2]
        
        seq_len = encoder_output.size(1)
        s = s.unsqueeze(0).repeat(seq_len, 1, 1) # s [seq_len, batch_size, decoder_hidden_num], 扩充、准备与encode_out进行拼接
        energy = torch.tanh(self.attnCal(torch.cat((s, encoder_output.permute(1, 0, 2)), dim=2))) 
        # energy [seq_len, batch_size, decoder_hidden_num]
        attention = self.v(energy).squeeze(2) # attention [seq_len, batch_size]
        
        return F.softmax(attention, dim=0).transpose(0, 1) # [batch_size, seq_len]


class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_hidden_num, decoder_hidden_num, dropRate, atten: Atten, temperature=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_hidden_num = encoder_hidden_num
        self.decoder_hidden_num = decoder_hidden_num
        self.atten = atten
        self.embed_size = decoder_hidden_num
        self.temperature = temperature # 训练时为1.生成时变换 0 - 2
        
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.rnn_d = nn.GRU(self.embed_size + encoder_hidden_num * 2, decoder_hidden_num, 
                            num_layers=1, bidirectional=False, batch_first=True)
        self.fc_out = nn.Linear((encoder_hidden_num * 2) + decoder_hidden_num + self.embed_size, vocab_size)
        self.drop = nn.Dropout(dropRate)
        
        self.softmax = nn.LogSoftmax(dim=-1)
    
    # 注意：加入attention机制后，逐个解码
    def forward(self, decoder_input, s, encoder_output):
        # decoder_input = [batch_size]
        # s = [batch_size, decoder_hidden_num]
        # encoder_output = [batch_size, seq_len, encoder_hidden_num * 2]
        decoder_input = decoder_input.unsqueeze(1) # [batch_size, 1]
        decoder_input = self.drop(self.embedding(decoder_input)) # [batch_size, 1, embed_size]
        
        a = self.atten(s, encoder_output).unsqueeze(1) # a [batch_size, 1, seq_len]
        c = torch.bmm(a, encoder_output) # c [batch_size, 1, encoder_hidden_num * 2]
        # 合并 & 输入RNN中
        rnn_input = torch.cat((decoder_input, c), dim = 2) # [batch_size, 1, self.embed_size + encoder_hidden_num * 2]
        decoder_output, decoder_hidden = self.rnn_d(rnn_input, s.unsqueeze(0)) # 此状态下，decoder_output, decoder_hidden一致
        # 将decoder_input、decoder_output(当前时刻的s)以及c拼在一起
        decoder_output = decoder_output.squeeze(1)
        c = c.squeeze(1)
        final_pred = self.fc_out(torch.cat((decoder_output, c, decoder_input.squeeze(1)), dim = 1)) # [batch_size, vocab_size]
        final_pred = final_pred / self.temperature # 实现温度采样
        
        return self.softmax(final_pred), decoder_hidden.squeeze(0) # 返回当前时刻的s
        

class AttenSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
    
    def forward(self, src, trg):
        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)  # 存储decoder的所有输出
        
        enc_output, s = self.encoder(src, None)
        
        # first input to the decoder is the <sos> tokens
        dec_input = trg[:, 0]  # target的第一列全是<SOS> -- decoder input的第一列

        for t in range(0, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)
            # 存储每个时刻的输出
            outputs[t] = dec_output
            # 用TeacherForce机制
            teacher_force = random.random() < self.teacher_forcing_ratio
            # 获取预测值
            top1 = dec_output.argmax(dim=1)
            if (t+1)!=trg_len:
                dec_input = trg[:, t+1] if teacher_force else top1

        return outputs.permute(1, 0, 2)
