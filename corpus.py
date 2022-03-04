#!/usr/bin/python3

# 读取文本文件，
# 文本清理
# 去除无用字符
# 输出语料 

corpus_file = '/home/fred/Documents/dev/corpus_tool/corpus/txt_133.txt'

def text_clean(corpus_file: str) -> str:
    """
    输入语料库文件地址，输出清洗后的语料文本
    """    
    text = ''
    with open(corpus_file, 'r') as f:
        while l := f.readline():
            if l != '\n':
                text += l[:-1].strip()
    return text

txt = text_clean(corpus_file)
print(txt)
