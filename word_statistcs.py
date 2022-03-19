#!/usr/bin/python3
import os
import jieba
from collections import Counter

CORPUS_PATH = '/home/fred/Documents/dev/taurus/corpus/'
PROJ_PATH = '/home/fred/Documents/dev/taurus/'

def work_flow_1():
    file_list = os.listdir(CORPUS_PATH)
    # 分词结果收入words当中，然后用word_dict进行分词
    words = []
    for f in file_list:
        # 读文件
        file_name = os.path.join(CORPUS_PATH, f)
        if os.path.getsize(file_name) > 1:
            with open(file_name, 'r') as r:
                txt = r.readlines()
                context = '\n'.join(txt)
            tmp = jieba.cut(context)
            words += list(tmp)
    # 分词
    # 统计词频, ((w1, f1),(w2, f2),……)
    word_dict = Counter(words)
    dict_file = os.path.join(PROJ_PATH, 'wd_statistics.txt')
    with open(dict_file, 'w') as w:
        for item in enumerate(word_dict):
            w.write(f"{item[0]} {int(item[1])}\n")

work_flow_1()
