import pandas as pd
import torch

from TcrDesign.model import BERT
from TcrDesign.dataset import WordVocab
from . import beam_search
from TcrDesign.gru_generate import AttenSeq2Seq, Encoder, Encoder_both, Decoder, Atten

# 定义参数值
vocab_path = "TcrDesign/data/vocab_1mer.pkl"
pre_beta_path = "TcrDesign/weights/rnn/betaVJ/RNN_tcr.ep"
pre_alpha_path = "TcrDesign/weights/rnn/alphaVJ/RNN_tcr.ep"

# 基因列表
TRA_gene_path = "TcrDesign/data/data_model/rnn_generate_VJ/alpha_VJ_genes.csv"
TRB_gene_path = "TcrDesign/data/data_model/rnn_generate_VJ/beta_VJ_genes.csv"

# BERT模型的超参
hidden = 256
layers = 6
attn_heads = 4
d_ff_fold = 4

num_layer = 1
encoder_hidden_num = 256
decoder_hidden_num = 256
encoder_num_layer = 1
dropout_rate = 0.3
teacher_forcing_ratio = 1
temperature = 1

# 有81个beta VJ基因，115个alpha VJ基因，5是为特殊字符预留
beta_VJ_num = 81 + 5
alpha_VJ_num = 115 + 5

cdr3_max_len = 22

def generate_VJ(cdr3, generate_num=10, with_cuda=True, is_alpha=False, batch_mode=False):
    if generate_num > 100:
        print("The max number of generate VJs is 100, set generate_num to 100!")
        generate_num = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Building model")
    vocab = WordVocab.load_vocab(vocab_path)
    Cbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    encoder = Encoder(Cbert, encoder_hidden_num, encoder_num_layer, decoder_hidden_num)
    atten = Atten(encoder_hidden_num, decoder_hidden_num)
    decoder = Decoder(alpha_VJ_num if is_alpha else beta_VJ_num, encoder_hidden_num, decoder_hidden_num, dropout_rate, atten, temperature)
    model = AttenSeq2Seq(encoder, decoder, device, teacher_forcing_ratio)
    if is_alpha:
        model.load_state_dict(torch.load(pre_alpha_path, map_location=device)) # 加载模型权重
    else:
        model.load_state_dict(torch.load(pre_beta_path, map_location=device))
    model.to(device)
    model.eval()
    
    # VJ mapping list
    if is_alpha:
        TR_genes = pd.read_csv(TRA_gene_path)["genes"].tolist()
    else:
        TR_genes = pd.read_csv(TRB_gene_path)["genes"].tolist()
    
    print("Starting to Generate")
    if batch_mode:
        encoder_input = [[vocab.sos_index] + [vocab.stoi[aa] for aa in one_cdr3] + [vocab.eos_index] + \
            [vocab.pad_index] * (cdr3_max_len - len(one_cdr3) - 2) for one_cdr3 in cdr3]
        encoder_input = torch.tensor(encoder_input)
    else:
        encoder_input = [vocab.sos_index] + [vocab.stoi[aa] for aa in cdr3] + [vocab.eos_index] + \
            [vocab.pad_index] * (cdr3_max_len - len(cdr3) - 2)
        encoder_input = torch.tensor(encoder_input).unsqueeze(0)
    
    decoded_VJs = beam_search(num_beams=generate_num, max_length=3, vocab=vocab, decoder=model, has_encoder=True, encoder_input=encoder_input, 
                              vocab_size=alpha_VJ_num if is_alpha else beta_VJ_num, decoder_input_prefix=None, with_cuda=with_cuda)
    
    decoded_VJs_list = []
    for vj in decoded_VJs:
        tmp_vj = [TR_genes[gene_id-5] for gene_id in vj]
        tmp_vj = tmp_vj[1:]
        decoded_VJs_list.append(tmp_vj)
    
    print("<<<Generate Done>>>")
    
    return decoded_VJs_list
