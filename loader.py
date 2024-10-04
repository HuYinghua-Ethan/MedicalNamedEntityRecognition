# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
rstrip() 和 strip() 是 Python 字符串对象的两个方法
主要区别在于它们去除空白字符的位置：

rstrip():
只去除字符串右侧（末尾）的空白字符（如空格、换行符等）。

strip():
同时去除字符串两端（即开头和末尾）的空白字符。

最长的句子的长度为311
这里设置截取300个字符


"""


class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        sentence = []
        labels = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                if not line:
                    continue
                char = line[0]
                if not char:
                    continue
                label = line[-1]
                sentence.append(char)
                labels.append(self.schema[label])
                if char in ['。','?','!','！','？']:  # 作为是句子的结束标志
                    # print(sentence)
                    # print(len(sentence))
                    # print(labels)
                    # input()
                    self.sentences.append("".join(sentence))
                    # print(self.sentences)
                    # input()
                    # if len(sentence) >= 300:
                    #     print("超过了max_seq_length")
                    #     print(sentence)
                    #     print(len(sentence))
                    #     input()
                    # self.sentences.append("".join(sentence))
                    # print(self.sentences)
                    # input()
                    input_ids = self.encode_sentence(sentence)
                    labels = self.padding(labels, -1)
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                    # print(self.data)
                    # input()
                    sentence = []
                    labels = []
        return
                    
                    

    def encode_sentence(self, sentence, padding=True): 
        input_ids = []
        for char in sentence:
            input_ids.append(self.vocab.get(char, self.vocab["UNK"]))
        if padding:
            input_ids = self.padding(input_ids)
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_ids, pad_token=0):
        input_ids = input_ids[:self.config["max_seq_length"]]
        input_ids += [pad_token] * (self.config["max_seq_length"] - len(input_ids))
        return input_ids


    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  #0留给padding位置，所以从1开始
        return token_dict


    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]





# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    from config import Config
    dataset = MyDataset("./data/train.txt", Config)




