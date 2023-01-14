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

#运行完整的共情对话系统
###################################################################
#情感分类


#词典类
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
        # 获取对应的label
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


#对batch数据进行处理
def collate_fn(batch):


    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train=True):
    emo_dataset = emoDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(emo_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#################################################################################
#用户话语情感分类


#定义设备
def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


#情感分类
def emoclass(line):
    model = torch.load('bert_prompt/prompt_model.pkl',map_location="cpu")
    model.to(device())
    review = tokenlize(line)
    vocab_model = pickle.load(open("bert_prompt/vocab.pkl", "rb"))
    result = vocab_model.transform(review,sequence_max_len)
    data = torch.LongTensor(result).to(device())
    data=torch.reshape(data,(1,sequence_max_len)).to(device())
    output = model(data)
    
    # 获取最大值的位置,[batch_size,1]
    pred = output.data.max(1, keepdim=True)[1]  
    return pred.item()
    
    
    
    
#################################################################################    
#回复情绪预测（情绪知识图谱查询）

def query(inn):
    graph = Graph('http://172.22.16.250:7474',auth=("neo4j", " "))#输入密码
    inn2=""
    if inn==0:
        inn2="悲伤"
    elif inn==1:
        inn2="高兴"
    elif inn==2:
        inn2="喜欢"
    elif inn==3:
        inn2="愤怒"
    elif inn==4:
        inn2="恐惧"
    elif inn==5:
        inn2="惊奇"
    elif inn==6:
        inn2="厌恶"

    mm = graph.run("MATCH (n:emo { name: '"+inn2+"' })-->(p:emo) RETURN p.num").data()
    nn=mm[0]

    kk=nn['p.num']
    return kk

    
#################################################################################
#共情对话生成

PAD = '[PAD]'
pad_id = 0


#参数设置
def set_interact_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False)
    parser.add_argument('--temperature', default=1, type=float, required=False)
    parser.add_argument('--topk', default=8, type=int, required=False)
    parser.add_argument('--topp', default=0, type=float, required=False)
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False)
    parser.add_argument('--log_path', default='data/interacting_mmi.log', type=str, required=False)
    parser.add_argument('--voca_path', default='vocabulary/vocab_small.txt', type=str, required=False)
    parser.add_argument('--dialogue_model_path', default='dialogue_model/', type=str, required=False)
    parser.add_argument('--mmi_model_path', default='mmi_model/', type=str, required=False)
    parser.add_argument('--save_samples_path', default="history/", type=str, required=False)
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=25)
    parser.add_argument('--max_history_len', type=int, default=5)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


#保存日志
def create_logger(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


#nucleus (top-p) 采样 
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:


        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for i in range(logits.size(0)): # logits.size(0)来获取logits的元素个数
            logit = logits[i]
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value


    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将指数向右移动，以保持第一个token在阈值之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def resp(inn):
    args = set_interact_args()
    logger = create_logger(args)
    # 当使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    
    # 对话model
    dialogue_model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    dialogue_model.to(device)
    dialogue_model.eval()
    
    # 互信息mmi model
    mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model_path)
    mmi_model.to(device)
    mmi_model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/mmi_samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    # print('开始聊天，输入CTRL + Z以退出')

    #输入话语
    text = inn
    
    #调用情感分类器
    bb = emoclass(text)
     
    #当前对话加入对话历史
    fo = open("history/history.txt", "a")
    fo.write(text+"\n")
    fo.close()

     
    if args.save_samples_path:
        samples_file.write("user:{}\n".format(text))
    history.append(tokenizer.encode(text))
    # 每个input以[CLS]为开头
    input_ids = [tokenizer.cls_token_id]
    for history_id, history_utr in enumerate(history[-args.max_history_len:]):
        input_ids.extend(history_utr)
        input_ids.append(tokenizer.sep_token_id)
    # 用于批量生成response，维度为(batch_size,token_len)
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]

    curr_input_tensors = torch.tensor(input_ids).long().to(device)
    generated = []  # 二维数组，维度为(生成的response的最大长度，batch_size)，generated[i,j]表示第j个response的第i个token的id
    finish_set = set()  # 标记是否所有response均已生成结束，若第i个response生成结束，即生成了sep_token_id，则将i放入finish_set
    
    
    # 最多生成max_len个token
    for _ in range(args.max_len):
        outputs = dialogue_model(input_ids=curr_input_tensors)
        next_token_logits = outputs[0][:, -1, :]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for index in range(args.batch_size):
            for token_id in set([token_ids[index] for token_ids in generated]):
                next_token_logits[index][token_id] /= args.repetition_penalty
        next_token_logits = next_token_logits / args.temperature
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token

        for i in range(len(next_token_logits)):
            next_token_logits[i][tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
        
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
        for index, token_id in enumerate(next_token[:, 0]):
            if token_id == tokenizer.sep_token_id:
                finish_set.add(index)
        # 检验是否所有的response均已生成[SEP]
        finish_flag = True  # 是否所有的response均已生成[SEP]的token
        for index in range(args.batch_size):
            if index not in finish_set:  # response批量生成未完成
                finish_flag = False
                break
        if finish_flag:
            break
        generated.append([token.item() for token in next_token[:, 0]])
        # 将新生成的token与原来的token进行拼接
        curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=-1)
    candidate_responses = []  # 生成的所有候选response
    for batch_index in range(args.batch_size):
        response = []
        for token_index in range(len(generated)):
            if generated[token_index][batch_index] != tokenizer.sep_token_id:
                response.append(generated[token_index][batch_index])
            else:
                break
        candidate_responses.append(response)

    # mmi模型的输入
    if args.debug:
        print("candidate response:")
    samples_file.write("candidate response:\n")
    min_loss = float('Inf')
    min_loss2 = float('Inf')
    best_response = ""
    best_response2 = ""
    
    kkk=0
    for response in candidate_responses:
        mmi_input_id = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        mmi_input_id.extend(response)
        mmi_input_id.append(tokenizer.sep_token_id)
        for history_utr in reversed(history[-args.max_history_len:]):
            mmi_input_id.extend(history_utr)
            mmi_input_id.append(tokenizer.sep_token_id)
        mmi_input_tensor = torch.tensor(mmi_input_id).long().to(device)
        out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
        loss = out[0].item()
        if args.debug:
            text = tokenizer.convert_ids_to_tokens(response)
            print("{} loss:{}".format("".join(text), loss))
        samples_file.write("{} loss:{}\n".format("".join(text), loss))

        #情感选择
        if emoclass("".join(tokenizer.convert_ids_to_tokens(response))) == bb:
            kkk=kkk+1
            if loss < min_loss:
                best_response = response
                min_loss = loss
        else:
            if loss < min_loss:
                best_response2 = response
                min_loss2 = loss

    '''
        #情感选择 KG
        kcl=query(bb)
        if emoclass("".join(tokenizer.convert_ids_to_tokens(response))) == kcl:
            kkk=kkk+1
            if loss < min_loss:
                best_response = response
                min_loss = loss
        else:
            if loss < min_loss:
                best_response2 = response
                min_loss2 = loss

    '''
    
    
    if kkk==0:
        best_response = best_response2
    history.append(best_response)
    text = tokenizer.convert_ids_to_tokens(best_response)
    #print("聊天机器人:" + "".join(text))
    outt="".join(text)
    if args.save_samples_path:
        samples_file.write("chatbot:{}\n".format("".join(text)))
    return outt


########################################################################################

#flask 后端

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return resp(str(userText))
       
##########################################################################################


if __name__ == '__main__':
#对话历史清除
    fo = open("history/history.txt", "w")
    fo.close()
    #flask运行
    app.run(host='0.0.0.0',port=10200)
