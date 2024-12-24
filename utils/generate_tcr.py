import torch

from TcrDesign.model import BERT
from TcrDesign.dataset import WordVocab
from . import beam_search
from TcrDesign.gru_generate import AttenSeq2Seq, Encoder, Encoder_both, Decoder, Atten

# 定义参数值
vocab_path = "TcrDesign/data/vocab_1mer.pkl"
pre_beta_path = "TcrDesign/weights/rnn/beta/RNN_tcr.ep"
pre_alpha_path = "TcrDesign/weights/rnn/beta_alpha/RNN_tcr.ep"

# BERT模型的超参
hidden = 256
layers = 6
attn_heads = 4
d_ff_fold = 4

antigen_max_len = 23
cdr3_max_len = 22

num_layer = 1
encoder_hidden_num = 256
decoder_hidden_num = 256
encoder_num_layer = 1
dropout_rate = 0.3
teacher_forcing_ratio = 1

# 采用beam_search生成时，pad和eos是后加的
# 如果batch生成的长度一样，就不会加eos和pad，否则加上pad和eos
def generate_betaTCRs(antigen, generate_num=1000, max_len=20, with_cuda=True,
                      temperature = 1):
    if generate_num > 50000:
        print("The max number of generate TCRs is 50000, set generate_num to 50000!")
        generate_num = 50000
    
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Building model")
    vocab = WordVocab.load_vocab(vocab_path)
    Pbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    encoder = Encoder(Pbert, encoder_hidden_num, encoder_num_layer, decoder_hidden_num)
    atten = Atten(encoder_hidden_num, decoder_hidden_num)
    decoder = Decoder(len(vocab), encoder_hidden_num, decoder_hidden_num, dropout_rate, atten, temperature)
    model = AttenSeq2Seq(encoder, decoder, device, teacher_forcing_ratio)
    model.load_state_dict(torch.load(pre_beta_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Starting to Generate")
    encoder_input = [vocab.sos_index] + [vocab.stoi[aa] for aa in antigen] + [vocab.eos_index] + \
        [vocab.pad_index] * (antigen_max_len - len(antigen) - 2)
    encoder_input = torch.tensor(encoder_input).unsqueeze(0)
    
    decoded_tcrs = beam_search(num_beams=generate_num, max_length=max_len, vocab=vocab, decoder=model, has_encoder=True, 
                               encoder_input=encoder_input, decoder_input_prefix=None, with_cuda=with_cuda)
    
    decoded_tcrs_list = list()
    for tcr in decoded_tcrs:
        tmp_tcr = [vocab.itos[aa] for aa in tcr]
        del tmp_tcr[0]
        try:
            del tmp_tcr[tmp_tcr.index('<eos>'):]
            tmp_tcr = ''.join(tmp_tcr)
        except:
            tmp_tcr = ''.join(tmp_tcr)
        if tmp_tcr.endswith('GF'): # 由于IEDB中的数据混乱，以GF结尾的TCR是多余的，需要去除GF即可
            tmp_tcr = tmp_tcr[:-2]
        decoded_tcrs_list.append(tmp_tcr)
    
    print("<<<Generate Done>>>")
    
    return decoded_tcrs_list


def generate_alphaTCRs(antigen, bCDR3, generate_num=1000, max_len=20, with_cuda=True,
                       temperature = 0.8):
    if generate_num > 50000:
        print("The max number of generate TCRs is 50000, set generate_num to 50000!")
        generate_num = 50000
        
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Building model")
    vocab = WordVocab.load_vocab(vocab_path)
    Pbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    Cbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    encoder = Encoder_both(Pbert, Cbert, encoder_hidden_num, encoder_num_layer, decoder_hidden_num, antigen_max_len)
    atten = Atten(encoder_hidden_num, decoder_hidden_num)
    decoder = Decoder(len(vocab), encoder_hidden_num, decoder_hidden_num, dropout_rate, atten, temperature)
    model = AttenSeq2Seq(encoder, decoder, device, teacher_forcing_ratio)
    model.load_state_dict(torch.load(pre_alpha_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Starting to Generate")
    encoder_input = [vocab.sos_index] + [vocab.stoi[aa] for aa in antigen] + [vocab.eos_index] + \
        [vocab.pad_index for _ in range(antigen_max_len - len(antigen) - 2)] + \
            [vocab.sos_index] + [vocab.stoi[aa] for aa in bCDR3] + [vocab.eos_index] + \
                [vocab.pad_index for _ in range(cdr3_max_len - len(bCDR3) - 2)]
    encoder_input = torch.tensor(encoder_input).unsqueeze(0)
    
    decoded_tcrs = beam_search(num_beams=generate_num, max_length=max_len, vocab=vocab, decoder=model, has_encoder=True, 
                               encoder_input=encoder_input, decoder_input_prefix=None, with_cuda=with_cuda)
    
    decoded_tcrs_list = list()
    for tcr in decoded_tcrs:
        tmp_tcr = [vocab.itos[aa] for aa in tcr]
        del tmp_tcr[0]
        try:
            del tmp_tcr[tmp_tcr.index('<eos>'):]
            tmp_tcr = ''.join(tmp_tcr)
        except:
            tmp_tcr = ''.join(tmp_tcr)
        if tmp_tcr.endswith('GF'): # 由于IEDB中的数据混乱，以GF结尾的TCR是多余的，需要去除GF即可
            tmp_tcr = tmp_tcr[:-2]
        decoded_tcrs_list.append(tmp_tcr)
    
    print("<<<Generate Done>>>")
    
    return decoded_tcrs_list
