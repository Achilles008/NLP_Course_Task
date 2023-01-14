# -*-coding:utf-8-*-
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from ERGMtrain import create_model
import torch.nn.functional as F
from torch.optim import Adam
import copy
import pickle
import re
import zipfile
import jieba
import pandas as pd
import sys
import requests
import time
import urllib.parse
import hashlib
from flask import Flask, render_template, request
from py2neo import Graph, Node, Relationship, NodeMatcher
import csv
class Vocab:
    UNK_TAG = "<UNK>"  # 表示未知字符
    PAD_TAG = "<PAD>"  # 填充符
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # 保存词语和对应的数字
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 统计词频

    #接受句子，统计词频
    def fit(self, sentence):

        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  # 所有的句子fit之后，self.count就有了所有词语的词频

    #根据条件构造 词典
    def build_vocab(self, min_count=1, max_count=None, max_features=None):

        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # 每次word对应一个数字

        # 把dict进行翻转
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    #把句子转化为数字序列
    def transform(self, sentence, max_len=None):

        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD

        return [self.dict.get(i, 1) for i in sentence]


    #把数字序列转化为字符
    def inverse_transform(self, incides):

        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)

class emoDataset(Dataset):
    def __init__(self, train=True):
        if train == True:
            url = 'train.xlsx'
        else:
            url = "test.xlsx"
        data = pd.read_excel(url)
        sentence = data.get('review')
        label = data.get('label')
        self.sentence_list=sentence
        self.label_list=label



    def __getitem__(self, idx):
        line_text=self.sentence_list[idx]
        # 从txt获取并分词
        review = tokenlize(str(line_text))
        # 获取评论对应的label
        label = int(self.label_list[idx])
        return review, label

    def __len__(self):
        return len(self.sentence_list)
        

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=200, padding_idx=voc_model.PAD).to()
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.1)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出

        # 进行全连接
        out_fc1 = self.fc1(out)
        # 进行relu
        out_fc1_relu = F.relu(out_fc1)

        # 全连接
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)

train_batch_size = 512
test_batch_size = 128

voc_model = pickle.load(open("bert_prompt/vocab.pkl", "rb"))
sequence_max_len = 100


def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = re.sub("|".join(fileters), "", sentence)
    sentence=jieba.cut(sentence,cut_all=False)
    sentence=' '.join(sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]
    result=movestopwords(result)
    return result

    
#停用词
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子去除停用词
def movestopwords(sentence):
    stopwords = stopwordslist('stopwords.txt')  # 这里加载停用词的路径
    outstr = []
    for word in sentence:
        if word not in stopwords:
            if word != '\t' and '\n':
                outstr.append(word)
                # outstr += " "
    return outstr



def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train=True):
    emo_dataset = emoDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(emo_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

