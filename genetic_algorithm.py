import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GAfunctions as GA
gongjian=[1,2,3,4,5,6]
machine=[1,2,3,4,5]
seq=[[1,2,4,1,3],[5,4,2,3,1],[2,4,1,5,4,1],[3,2,1,2],[2,3,4,3],[2,4,5,1,3,2]]
processtime=[[20,30,40,60,10],[70,80,30,90,40],[60,70,10,20,70,30],\
             [10,10,20,20],[30,50,10,10],[30,30,60,10,20,50]]
dt=[[350,400],[450,500],[450,500],[400,450],[400,450],[350,400]]
punish=[[4,4],[5,5],[5,5],[2,3],[2,3],[4,4]]
size=100
rd.seed(3) 
origin = []
pc=0.8
pm=0.05
alpha = 0.01
top=10
iteration_max=50
iteration=0
n_iter=[]
zf=1/alpha
avg_score=[]

# construct initial group of genes
for i in range(len(gongjian)):
    origin=origin+[i+1]*len(seq[i])

population=pd.DataFrame()
for j in range(size):
    rd.shuffle(origin)
    population[j]=origin

# calculate fitness score of current group
for gene in population:
    gen_score = {}
    sum_score=0
    for gene in population:
        gen_score[gene]=GA.score(GA.schedule(population[gene])[1])
        sum_score+=gen_score[gene]
avg_score.append(sum_score/size)

while (iteration<=iteration_max) & (avg_score[iteration]!=zf):

    #duplicate top 10% genes to next generation
    next_population=GA.duplicate(population,gen_score,top)

    # choose rest 90 genes by roulette principle
    gen_select=[]
    temp_pop=pd.DataFrame()
    for k in range(size-top):
        gen_select.append(GA.dice(gen_score))
        temp_pop[k]=population[gen_select[k]]

    # cross at a probability of pc
    index=[ind for ind in range(size-top)]
    rd.shuffle(index)
    num_group=(size-top)//2
    group1=index[:num_group]
    group2=index[num_group:]
    for pair in range(num_group):
        temp_pop[group1[pair]],temp_pop[group2[pair]]=\
        GA.cross(temp_pop[group1[pair]],temp_pop[group2[pair]])

    # mutate at a probability of pm
    for k2 in range(size-top):
        next_population[k2+10]=temp_pop[k2]
    for k3 in range(size):
        next_population[k3]=GA.mutate(next_population[k3])
    population=next_population

    # calculate fitness score for new group
    for gene in population:
        gen_score = {}
        sum_score=0
        for gene in population:
            gen_score[gene]=GA.score(GA.schedule(population[gene])[1])
            sum_score+=gen_score[gene]
    avg_score.append(sum_score/size)
    n_iter.append(iteration)
    iteration+=1
    
n_iter.append(iteration)
print(avg_score)
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.set(title='Result',ylabel='avg.fitness', xlabel='iterations')
plt.plot(n_iter,avg_score)
plt.grid(True)

zscore=[f**(-1)-alpha for f in avg_score]
fig2=plt.figure(2)
bx=fig2.add_subplot(111)
bx.set(title='Result',ylabel='avg.obj.fun', xlabel='iterations')
plt.plot(n_iter,zscore)
plt.show()

rank=sorted(gen_score.items(),key=lambda x:x[1],reverse=True)
solution=population[rank[0][0]]
machine_time,piece_time=GA.schedule(solution)
fsolution=GA.score(piece_time)
zsolution=1/fsolution-alpha
print(machine_time)
print(piece_time)
print(iteration)
print(fsolution)
print(zsolution)
