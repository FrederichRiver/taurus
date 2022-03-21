#!/usr/bin/python3

import os
import re

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFTextExtractionNotAllowed, PDFPage

from word_statistcs import PROJ_PATH

# 读取文本文件，
# 文本清理
# 去除无用字符
# 输出语料 

PDF_PATH = '/data1/file_data/report/industry_report/'
TOTAL_CLASS = ['医药商业', '医疗服务', '医疗器械', '医疗行业', '医药制造']
CORPUS_CLASS = ['医药制造', ]
PROJ_PATH = "/home/fred/Documents/dev/taurus"
corpus_file = '/home/fred/Documents/dev/corpus_tool/corpus/txt_133.txt'
GBK = 'GBK'
GB = 'GB2312'

def get_file_list(p: str) -> list:
    return os.listdir(p)

def codefilter(w: str):
    """
    过滤全角符号
    """
    if ord(w) < 127:
        return w
    elif w.encode('gbk') < b'\xa9\xfe':
        return ''
    else:
        return w

def title_clean(title: str) -> str:
    newtitle = ''
    for w in title:
        newtitle += codefilter(w)
    return newtitle

def remove_cw(txt: str) -> str:
    txt = txt.replace(u'\n', '')
    txt = txt.replace(' ', '')
    return txt

def pdf2text(pdf_file: str) -> list:
    """
    pdf文件转换为文本
    """

    # 用文件对象来创建一个pdf文档分析器
    praser = PDFParser(fp = open(pdf_file, 'rb'))
    # 创建一个PDF文档
    doc = PDFDocument(praser)
    # 连接分析器 与文档对象
    praser.set_document(doc)

    corpus = []
    section = []
    line_size = 1
    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        # raise PDFTextExtractionNotAllowed
        tmp_name = pdf_file.split('/')[-1]
        print(f"{tmp_name}: PDFTextExtractionNotAllowed")
    else:
        # 创建PDf 资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # 循环遍历列表，每次处理一个page的内容
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)                        
            # 接受该页面的LTPage对象
            layout = device.get_result()
            # 这里layout是一个LTPage对象，里面存放着这个 page 解析出的各种对象
            # 包括 LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等                            
            for x in layout:
                # print(x)
                if isinstance(x, LTTextBox):
                    line = x.get_text().strip()
                    corpus.append(line)
                    # line_size = max(len(line), line_size)
    line_size = 60    
    rex = re.compile(r'.+[!|！|.|。|?|？]$')
    sec = ''
    for l in corpus:
        if l:
            sec += l
            if len(l) < (0.65 * line_size) or re.match(rex, l):
                sec = remove_cw(sec)
                section.append(sec)
                sec = ''
                
    return section

def txt_clean(corpus: str) -> str:
    tmp = ''
    cw = set(['\n', ' '])
    for w in corpus:
        if w not in cw:
            tmp += w
    return tmp


def save_txt(txt_file: str, txt: list, th=20):
    with open(txt_file, 'w') as f:
        for item in txt:
            if len(item) > th:
                f.write(f"{item}\n")
    return 0 if os.path.getsize(txt_file) == 0 else 1

def pdf_struc_view(pdf_file: str):
    praser = PDFParser(fp = open(pdf_file, 'rb'))
    # 创建一个PDF文档
    doc = PDFDocument(praser)
    # 连接分析器 与文档对象
    praser.set_document(doc)

    corpus = []
    section = []
    line_size = 1
    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDf 资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        print(f"Title: {pdf_file}")
        i = 1
        # 循环遍历列表，每次处理一个page的内容
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)                        
            # 接受该页面的LTPage对象
            layout = device.get_result()
            # 这里layout是一个LTPage对象，里面存放着这个 page 解析出的各种对象
            # 包括 LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等                            
            print(f"Page{i}")
            i += 1
            for x in layout:
                print(x)

def work_flow_1():
    """
    文本前处理流程
    """
    # 获取
    txt_path = "/home/fred/Documents/dev/taurus/corpus"
    log_file = os.path.join(PROJ_PATH, 'log_file')
    pdf_path_list = [os.path.join(PDF_PATH, c) for c in CORPUS_CLASS]
    for p in pdf_path_list:
        file_list = get_file_list(p)
        for pdf_file in file_list:
            pdf = os.path.join(p, pdf_file)
            try:    
                txt_title = pdf_file[:-3] + 'txt'
                txt_file_name = title_clean(txt_title)
                txt_file = os.path.join(txt_path, txt_file_name)
                tmp_txt = pdf2text(pdf)
                txt = []
                for sec in tmp_txt:
                    txt.append(txt_clean(sec))
                if save_txt(txt_file, txt):
                    print(f'{pdf_file}')
                else:
                    with open(log_file, 'a') as f:
                        f.write(f"{pdf}\n")
            except:
                with open(log_file, 'a') as f:
                        f.write(f"{pdf}\n")


def work_flow_2():
    # 获取
    pdf_path = os.path.join(PDF_PATH, '医药商业')
    pdf = os.path.join(pdf_path, "2022年医药行业策略报告：回本溯源，看好药品行业的长期投资机会-2021-12-09.pdf")
    pdf_struc_view(pdf)

work_flow_1()


# work_flow_2()

