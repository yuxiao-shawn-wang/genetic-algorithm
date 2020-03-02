import random as rd
import numpy as np
import pandas as pd
size=100
pc=0.8
pm=0.05
seq=[[1,2,4,1,3],[5,4,2,3,1],[2,4,1,5,4,1],[3,2,1,2],[2,3,4,3],[2,4,5,1,3,2]]
processtime=[[20,30,40,60,10],[70,80,30,90,40],[60,70,10,20,70,30],[10,10,20,20],\
             [30,50,10,10],[30,30,60,10,20,50]]
dt=[[350,400],[450,500],[450,500],[400,450],[400,450],[350,400]]
punish=[[4,4],[5,5],[5,5],[2,3],[2,3],[4,4]]

# scheduling function
def schedule(genes):
    count = {}
    M = [[0],[0],[0],[0],[0]]
    G = [[0],[0],[0],[0],[0],[0]]
    for gene in genes:
        if gene not in count.keys():
            count[gene] = 1
        else:
            count[gene] += 1

        m=seq[gene-1][count[gene]-1]    # current machine number
        t=processtime[gene-1][count[gene]-1]    # current process time
        st=max(M[m-1][-1],G[gene-1][count[gene]-1])    # machine start time
        M[m-1].append(st+t)
        G[gene-1].append(st+t)
    return M,G

# fitness score
def score(G):
    z=0
    alpha = 0.01
    f = []
    for g in range(6):
        z += punish[g][0]*max(0,dt[g][0]-G[g][-1])+\
        punish[g][1]*max(0,G[g][-1]-dt[g][1])
    f = 1/(alpha+z)
    return f

# duplicate best genes
def duplicate(Pop,Pop_score,topN):
    # sort all genes in descending order
    rank=sorted(Pop_score.items(),key=lambda x:x[1],reverse=True)
    top_index = []
    Pop_copy=pd.DataFrame()
    for i in range(topN):
        top_index.append(rank[i][0])
        Pop_copy[i]=Pop[top_index[i]]
    return Pop_copy

# choose one gene using roulette principle
def dice(score_board):
    index = list(score_board.keys())
    score = list(score_board.values())
    sumscore = sum(score)
    rndPoint = rd.uniform(0, sumscore)
    accumulator = 0.0
    round = -1
    for val in score:
        accumulator += val
        round += 1
        if accumulator >= rndPoint:
            return index[round]

# cross function for genes
def cross(gen1,gen2):
    global size,pc
    rnd=rd.uniform(0,1)
    if rnd>pc:
        return gen1,gen2
    else:
        temp1=gen1
        temp2=gen2
        index=[1,2,3,4,5,6]
        rd.shuffle(index)
        g1 = index[:3]
        x = [0 for xi in range(30)]
        y = [0 for yi in range(30)]
        for k in range(30):
            if temp1[k] in g1:
                x[k]=temp1[k]
                temp1[k]=-1
            
            if temp2[k] in g1:
                y[k]=temp2[k]
                temp2[k]=-1
        l1 = 0
        l2 = 0
        while l1 < len(temp2):
            if temp2[l1]<0:
                l1+=1
            else:
                num1=x.index(0)
                x[num1]=temp2[l1]
                l1+=1

        while l2 < len(temp1):
            if temp1[l2]<0:
                l2+=1
            else:
                num2=y.index(0)
                y[num2]=temp1[l2]
                l2+=1
        return x,y

# mutate function for genes
def mutate(before):
    if rd.uniform(0, 1) < pm:
        x,y = rd.sample(range(30),2)
        a = before[x]
        b = before[y]
        after = before.copy()
        after[x] = b
        after[y] = a
        return after
    else:
        return before
