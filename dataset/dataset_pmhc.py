from torch.utils.data import Dataset
import tqdm
import torch
import random

# 数据预处理类
class BERTDataset_pMHC(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, mask_freq=0.2, encoding="utf-8", corpus_lines=None, on_memory=True, include_label=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.mask_freq = mask_freq
        self.include_label = include_label

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
    
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, item):
        mhc, peptide= self.lines[item][0], self.lines[item][1]
        if self.include_label:
            score = float(self.lines[item][2])
        # if score >= 0.4256252:
        #     mhc_random, mhc_label = self.random_word(mhc)
        #     peptide_random, peptide_label = self.random_word(peptide)
        # else:
        #     mhc_random = [self.vocab.stoi.get(aa, self.vocab.unk_index) for aa in mhc]
        #     peptide_random = [self.vocab.stoi.get(aa, self.vocab.unk_index) for aa in peptide]
        #     mhc_label = [0] * len(mhc_random)
        #     peptide_label = [0] * len(peptide_random)
        # mhc_random, mhc_label = self.random_word(mhc, mask_freq=0.1)
        # peptide_random, peptide_label = self.random_word(peptide, mask_freq=0.2)
        mhc_random, mhc_label = self.random_word(mhc, mask_freq=self.mask_freq)
        peptide_random, peptide_label = self.random_word(peptide, mask_freq=self.mask_freq)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        mhc = [self.vocab.sos_index] + mhc_random + [self.vocab.eos_index]
        peptide = peptide_random + [self.vocab.eos_index]

        mhc_label = [self.vocab.sos_index] + mhc_label + [self.vocab.eos_index]
        peptide_label = peptide_label + [self.vocab.eos_index]

        # seq_len即一个句子的最大长度，多的截断，少的补齐
        segment_label = ([1 for _ in range(len(mhc))] + [2 for _ in range(len(peptide))])[:self.seq_len]
        bert_input = (mhc + peptide)[:self.seq_len]
        bert_label = (mhc_label + peptide_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        if self.include_label:
            output = {"bert_input": bert_input,
                    "bert_label": bert_label,
                    "segment_label": segment_label,
                    "binding_score": score}
        else:
            output = {"bert_input": bert_input,
                    "bert_label": bert_label,
                    "segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}
    
    def random_word(self, sentence, mask_freq=0.2):
        # tokens = sentence.split() # split一下
        tokens = [s for s in sentence]
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < mask_freq: # 原先 0.2
                prob /= mask_freq

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(5, len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label  # tokens即经过mask处理后的index序列，output_label表示mask的标签
