# -*- coding: utf-8 -*-
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import csv
graph = Graph('http://localhost:7474', username='neo4j', password=' ')

graph.delete_all()
##################################
node1 = Node('emo',name = "高兴",num=1)
graph.create(node1)
node2 = Node('emo',name = "喜欢",num=2)
graph.create(node2)
node3 = Node('emo',name = "愤怒",num=3)
graph.create(node3)
node4 = Node('emo',name = "悲伤",num=4)
graph.create(node4)
node5 = Node('emo',name = "恐惧",num=5)
graph.create(node5)
node6 = Node('emo',name = "厌恶",num=6)
graph.create(node6)
node7 = Node('emo',name = "惊奇",num=7)
graph.create(node7)

pp1 = Relationship(node1, '共情', node1)
graph.create(pp1)
pp2 = Relationship(node2, '共情', node2)
graph.create(pp2)
pp3 = Relationship(node3, '共情', node3)
graph.create(pp3)
pp4 = Relationship(node4, '共情', node2)
graph.create(pp4)
pp5 = Relationship(node5, '共情', node2)
graph.create(pp5)
pp6 = Relationship(node6, '共情', node2)
graph.create(pp6)
pp7 = Relationship(node7, '共情', node1)
graph.create(pp7)

#####################################
with open('class.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)
with open('phrase1.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    data1 = list(reader)
for i  in  range(0,len(data)):
    relation = Node('phr',name = data[i][1])
    graph.create(relation)
    print(data[i][2])
    for j in range(0, len(data1)):
        if data1[j][1] == data[i][1]:
            jiedian = Node('ci', name=data1[j][0])
            graph.create(jiedian)
            qq = Relationship(jiedian, '属于', relation)
            graph.create(qq)

    if data[i][2] =="1":
        pp = Relationship(relation,'属于', node1)
        graph.create(pp)
    elif data[i][2] =="2":
        pp = Relationship(relation,'属于', node2)
        graph.create(pp)
    elif data[i][2] =="3":
        pp = Relationship(relation,'属于', node3)
        graph.create(pp)
    elif data[i][2] =="4":
        pp = Relationship(relation,'属于', node4)
        graph.create(pp)
    elif data[i][2] =="5":
        pp = Relationship(relation,'属于', node5)
        graph.create(pp)
    elif data[i][2] =="6":
        pp = Relationship(relation,'属于', node6)
        graph.create(pp)
    elif data[i][2] =="7":
        pp = Relationship(relation,'属于', node7)
        graph.create(pp)