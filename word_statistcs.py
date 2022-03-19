#!/usr/bin/python3
import os
import jieba
from collections import Counter
import psutil

CORPUS_PATH = '/home/fred/Documents/dev/taurus/corpus/'
PROJ_PATH = '/home/fred/Documents/dev/taurus/'

def isZh(w: str) -> bool:
    try:
        if w.encode('gbk') > b'\xaf\xa0':
            return True
        else:
            return False
    except:
        return False

def mem_usage() -> float:
    mem_available = psutil.virtual_memory().available
    mem_process = psutil.Process(os.getpid()).memory_info().rss
    return round(mem_process / mem_available, 2)

def get_dict(dict_file: str) -> dict:
    wd_dict = {}
    with open(dict_file, 'r') as f:
        while line := f.readline():
            line = line.replace(u'\n', '')
            if line:
                k, v = line.split(" ")
                if k:
                    wd_dict[k] = int(v)
    return wd_dict

def work_flow_1():
    """
    根据语料统计词频
    """
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
    # print(type(word_dict))
    dict_file = os.path.join(PROJ_PATH, 'wd_statistics.txt')
    with open(dict_file, 'w') as w:
        for k, v in word_dict.items():
            w.write(f"{k} {int(v)}\n")

def work_flow_2(th=10):
    """
    分类词汇
    """
    dict_file = os.path.join(PROJ_PATH, 'wd_statistics.txt')
    none_zh_dict = os.path.join(PROJ_PATH, 'nzh_dict.txt')
    onewd_dict = os.path.join(PROJ_PATH, 'oneword_dict.txt')
    zh_dict = os.path.join(PROJ_PATH, 'zh_dict.txt')
    n = open(none_zh_dict, 'w')
    zh = open(zh_dict, 'w')
    ow = open(onewd_dict, 'w')
    wd_dict = get_dict(dict_file)
    for k, v in wd_dict.items():
        if not isZh(k[0]):
            n.write(f"{k} {v}\n")
        else:
            if len(k) == 1:
                ow.write(f"{k} {v}\n")
            else:
                if int(v) > th:
                    zh.write(f"{k} {v}\n")

def work_flow_3():
    dict_file = os.path.join(PROJ_PATH, 'zh_dict.txt')
    wd_dict = get_dict(dict_file)
    dict_order(wd_dict)

def dict_order(wd_dict: dict) -> dict:
    # 获取词频序列
    wd_freq = set(wd_dict.values())
    new_dict = {}
    tmp = []
    # 词频序列倒序，似乎有错误
    for i in range(max(wd_freq) + 1, 0, -1):
        if i in wd_freq:
            tmp.append(i)
        i += 1
    # 冒泡排序
    for j in tmp:
        for k, v in wd_dict.items():
            if v > j:
                new_dict[k] = v
                wd_dict[k] = 0
    # 写入词典
    ordered_dict = os.path.join(PROJ_PATH, 'ordered_dict.txt')
    with open(ordered_dict, 'w') as f:
        for k, v in new_dict.items():
            f.write(f"{k} {v}\n")

# work_flow_1()
# work_flow_2()
work_flow_3()