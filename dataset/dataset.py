from torch.utils.data import Dataset
import tqdm
import torch
import random

# 数据预处理类
class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, mask_freq=0.2, encoding="utf-8", corpus_lines=None, on_memory=True, has_next=False):
        self.vocab = vocab
        self.seq_len = seq_len
        self.has_next = has_next
        self.mask_freq = mask_freq

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

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
        if self.has_next:
            t1, t2, is_next_label = self.random_sent(item)
            t1_random, t1_label = self.random_word(t1)
            t2_random, t2_label = self.random_word(t2)

            # [CLS] tag = SOS tag, [SEP] tag = EOS tag
            t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
            t2 = t2_random + [self.vocab.eos_index]

            t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
            t2_label = t2_label + [self.vocab.pad_index]

            # seq_len即一个句子的最大长度，多的截断，少的补齐
            segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
            bert_input = (t1 + t2)[:self.seq_len]
            bert_label = (t1_label + t2_label)[:self.seq_len]

            padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

            output = {"bert_input": bert_input,
                    "bert_label": bert_label,
                    "segment_label": segment_label,
                    "is_next": is_next_label}

            return {key: torch.tensor(value) for key, value in output.items()}
        
        else:
            t = self.lines[item][0]
            t_random, t_label = self.random_word(t)
            
            # [CLS] tag = SOS tag, [SEP] tag = EOS tag
            t = [self.vocab.sos_index] + t_random + [self.vocab.eos_index]
            t_label = [self.vocab.pad_index] + t_label + [self.vocab.pad_index]
            # t_label = [self.vocab.sos_index] + t_label + [self.vocab.eos_index]
            
            bert_input = t[:self.seq_len]
            bert_label = t_label[:self.seq_len]
            
            padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding)
            
            output = {"bert_input": bert_input,
                      "bert_label": bert_label}   
            
            return {key: torch.tensor(value) for key, value in output.items()}


    def random_word(self, sentence):
        # tokens = sentence.split() # split一下
        tokens = [s for s in sentence]
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < self.mask_freq: # 原先 0.2
                prob /= self.mask_freq

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

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
