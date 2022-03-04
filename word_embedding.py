#!/usr/bin/python3

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

def get_dict(dict_file: str, th=0):
    voc = []
    freq = []
    with open(dict_file, 'r') as d:
        while l := d.readline():
            w, f, _ = l.split(' ')
            voc.append(w)
            freq.append(int(f))
    return voc, freq

DICT_FILE = '/home/fred/Documents/dev/taurus/dict.txt'

v, f = get_dict(DICT_FILE)

import torch
from torch import nn

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
        self.noise_dist = noise_dist
        #定义词向量层
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        #词向量层参数初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    #输入词的前向过程
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    #目标词的前向过程
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    #负样本词的前向过程
    def forward_noise(self, size, N_SAMPLES):
        noise_dist = self.noise_dist
        #从词汇分布中采样负样本
        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors

from torch import tensor

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
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        #负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)
        #综合计算两类损失
        return -(out_loss + noise_loss).mean()