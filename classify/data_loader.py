#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os

def read_file(filename):
    """读取文件数据"""
    contents = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                contents.append(list(line))
            except:
                pass
    return contents

def read_vocab(vocab_dir):
    """读取词汇表"""
    vocab_file = open(vocab_dir, 'r', encoding='utf-8').readlines()
    words = list(map(lambda line: line.strip(),vocab_file))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['讨薪纠纷', '其他']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, max_length=600):
    """将文件转换为id表示"""
    contents = read_file(filename)

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return x_pad

