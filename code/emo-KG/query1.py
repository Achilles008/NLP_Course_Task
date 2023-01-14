# -*- coding: utf-8 -*-
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import csv
graph = Graph('http://172.22.16.250:7474',auth=("neo4j", " "))

inn = int(input("请输入："))
inn2=""
if inn==1:
    inn2="高兴"
elif inn==2:
    inn2="喜欢"
elif inn==3:
    inn2="愤怒"
elif inn==4:
    inn2="悲伤"
elif inn==5:
    inn2="恐惧"
elif inn==6:
    inn2="厌恶"
elif inn==7:
    inn2="惊奇"

mm = graph.run("MATCH (n:emo { name: '"+inn2+"' })-->(p:emo) RETURN p.num").data()
nn=mm[0]

kk=nn['p.num']
print(kk)