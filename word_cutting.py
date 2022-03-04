#!/usr/bin/python3

import jieba
from jieba import cut

sentence = '我爱北京天安门，天安门上太阳升。'

result = cut(sentence)

print('/'.join(result))