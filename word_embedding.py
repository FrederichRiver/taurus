#!/usr/bin/python3

import jieba
import os
import torch
from collections import Counter
from torch import nn, tensor


EMBEDDING_DIM = 128 #词向量维度
PRINT_EVERY = 100 #可视化频率
EPOCHES = 1000 #训练的轮数
# BATCH_SIZE = 5 #每一批训练数据大小
N_SAMPLES = 3 #负样本大小
WINDOW_SIZE = 5 #周边词窗口大小
FREQ = 5 #词汇出现频数的阈值
DELETE_WORDS = False #是否删除部分高频词
VOCABULARY_SIZE = 50000

# 建立词典
# 统计词频

# 读取字典

DICT_FILE = '/home/fred/Documents/dev/taurus/dict.txt'

def get_dict(dict_file: str, th=0):
    """
    从jieba分词的词典文件构造词典dict。
    """
    voc = []
    freq = []
    with open(dict_file, 'r') as d:
        while l := d.readline():
            w, f, _ = l.split(' ')
            voc.append(w)
            freq.append(int(f))
    return voc, freq

def generate_dict(corpus: str) -> dict:
    """
    通过语料生成词典dict
    """
    # d { w: c}
    d = {}
    # r is list of words
    r = list(jieba.cut(corpus))
    for item in r:
        if item in d.keys():
            d[item] += 1
        else:
            d[item] = 1
    return d

def merge_dict(d1: dict, d2: dict) -> dict:
    """
    合并两个字典
    """
    for k, v in d1.items():
        if k in d2.keys():
            d2[k] += v
        else:
            d2[k] = v
    return d2

def order_dict(d: dict, size: int) -> dict:
    new_d = {}
    m = max(list(d.values()))
    while len(new_d) < size:
        for k, v in d.items():
            if v > m:
                new_d[k] = v
                d.pop(k)
        m -= 1
    return new_d

class SkipGram(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, noise_dist):
        """
        n_vocab: dimension of vocab
        n_embed: dimension of word embedding
        the shape of word vector matrix: (n_vocab, n_embed)
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        # 负样本权重分布
        self.noise_dist = noise_dist
        #定义词向量层
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        #词向量层参数初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    #输入词的前向过程
    def forward_input(self, input_words: tensor) -> tensor:
        # (batch, n_embed)
        input_vectors = self.in_embed(input_words)
        return input_vectors
    #目标词的前向过程
    def forward_output(self, output_words: tensor) -> tensor:
        # (batch, n_embed)
        output_vectors = self.out_embed(output_words)
        return output_vectors
    #负样本词的前向过程
    def forward_noise(self, size: int, N_SAMPLES: int) -> tensor:
        noise_dist = self.noise_dist
        #从词汇分布中采样负样本
        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        # shape(size, n_samples, n_embed)
        noise_vectors = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    """
    loss function of negative-sampling.\n
    """
    def forward(self, input_vectors: tensor, output_vectors: tensor, noise_vectors: tensor):
        batch_size, embed_size = input_vectors.shape
        #将输入词向量与目标词向量作维度转化处理
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        #目标词损失
        test = torch.bmm(output_vectors, input_vectors)
        # (batch, 1, 1)
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        # (batch)
        out_loss = out_loss.squeeze()
        #负样本损失
        # (batch, n_sample, n_embed) (batch, n_embed, 1) = (batch, n_sample, 1)
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        # after squeeze, (batch, n_sample); after sum, (batch)
        noise_loss = noise_loss.squeeze().sum(1)
        #综合计算两类损失
        return -(out_loss + noise_loss).mean()

from torch.utils.data import Dataset

class WordEmbeddingDataset(Dataset):
    def __init__(self, data: list, word_freqs: list):
        """
        data : words in one-hot form.
        word_freqs: word freq in list.
        """
        super(WordEmbeddingDataset, self).__init__()
        # data的维度 (m, n) m为词数，n为词向量的维度
        self.data = torch.Tensor(data).long()  # 解码为词表中的索引
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率

    def __len__(self):
        # 共有多少个item
        return len(self.data)

    def __getitem__(self, idx):
        # 根据idx返回
        # center word in one-hot tensor (1, n_embedding)
        center_word = self.data[idx]  # 找到中心词
        # 对两个list求并集，WINDOW_SIZE是窗口宽度的一半。中心词前后各WINDOW_SIZE个词作为正样本
        pos_indices = list(range(idx - WINDOW_SIZE, idx)) + list(range(idx + 1, idx + WINDOW_SIZE + 1))  
        # 过滤，如果索引超出范围，则丢弃
        pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  
        # 周围单词 (n_pos_indices, n_embedding)
        pos_words = self.data[pos_indices]
        # multinomial按词频权重进行采样N_SAMPLES×pos_words.shape[0]个负样本。
        # neg_words 维度(N_SAMPLES * pos_words.shape[0], n_embedding)
        neg_words = torch.multinomial(self.word_freqs, N_SAMPLES * pos_words.shape[0], True)
        return center_word, pos_words, neg_words

def word_filter(words: list, th: int):
    word_count = Counter(words)
    result = [w for w in words if word_count[w]>th]
    return result


def count_word_freq(word_seq: list) -> dict:
    """
    word_seq是以词典定义的词序列，用count_word_freq方法用于统计词频以供生成负样本。
    """
    # {"word": count}
    int_word_counts = Counter(word_seq)
    # 语料中的总词数
    total_count = len(word_seq)
    # 计算词频
    word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
    return word_freqs


CORPUS = "经济规模达到114.4万亿元，人均GDP按年平均汇率折算达12551美元，超过世界人均GDP水平；预计占世界经济比重超过18%，对世界经济增长贡献率达到25%左右……开局之年，中国经济实力、社会生产力和综合国力都上了一个新台阶，这是一个实打实的“开门红”。"


def test():
    # d = generate_dict(CORPUS)
    dict_file = '/home/fred/Documents/dev/taurus/zh_dict.txt'
    jieba.load_userdict(dict_file)
    d = jieba.cut(CORPUS)
    print('/'.join(list(d)))

def train_word_vector(args: dict):
    # Read corpus
    from jieba import cut

    result = cut(CORPUS)
    r = list(result)
    # statistic word freqs
    wf = Counter(list(result))
    # print(wf)
    w = word_filter(r, 3)
    print(w)
    # Generate dataset
    # model definition
    # loss function definition
    # training process
    

if __name__ == '__main__':
    import sys
    # train_word_vector(sys.argv)
    test()