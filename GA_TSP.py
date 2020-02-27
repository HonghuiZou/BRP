import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy import linalg as la, inexact
import time


start = time.time()
# cities = np.array([
#         [256, 121],
#         [264, 715],
#         [225, 605],
#         [168, 538],
#         [210, 455],
#         [120, 400],
#         [96, 304],
#         [10,451],
#         [162, 660],
#         [110, 561],
#         [105, 473]
#         ])

cities=np.array([[41,94],[37,84],[54,67],[25,62],[7,64],[2,99],[68,58],[71,44],[54,62],[83,69], [64,60],[18,54],[22,60],[83,46],[91,38],[25,38],[24,42],[58,69],[71,71],[74,78], [87,76],[18,40],[13,40],[82,7],[62,32],[58,35],[45,21],[41,26],[44,35],[4,50] ])
city_count = len(cities)
Distance = np.zeros([city_count, city_count])
for i in range(city_count):
                for j in range(city_count):
                        Distance[i][j] = math.sqrt(
                                (cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2)
#种群数
count=300
#起点城市
origin = 0
#强者存活率
retain_rate=0.3
#弱者存活概率
random_rate=0.5
#个体改良次数
improve_count=2000
#变异率
mutation_rate=0.3
#迭代次数
item_time=1000
#路线总距离
def get_total_distance(x):
        distance=0
        distance+=Distance[origin][x[0]]
        distance += Distance[origin][x[-1]]
        for i in range(len(x)-1):
                distance+=Distance[x[i]][x[i+1]]
        return distance

#优质生成个体
def improve(x):
        i=0
        distance=get_total_distance(x)
        while i<improve_count:
                a=random.randint(1,len(x)-1)
                b=random.randint(1,len(x)-1)
                if a!=b:
                        new_x=x.copy()
                        t=new_x[a]
                        new_x[a]=new_x[b]
                        new_x[b]=t
                        new_distance=get_total_distance(new_x)
                        if new_distance<=get_total_distance(x):
                                x=new_x.copy()
                else:
                        continue
                i=i+1
        return x

#适应度
def selection(population):
        graded=[[get_total_distance(x),x] for x in population]
        graded=[x[1] for x in sorted(graded)]
        retain_length=int(len(graded)*retain_rate)
        parents=graded[:retain_length]
        for chromosome in graded[retain_length:]:
                if random.random()<random_rate:
                        parents.append(chromosome)
        return parents

#交叉繁殖
def crossover(parents):
        #还需要的子代个数，保持总数count稳定
        target_count=count-len(parents)
        #子代列表
        children=[]
        while len(children)<target_count:
                #父母代索引
                male_index=random.randint(0,len(parents)-1)
                female_index=random.randint(0,len(parents)-1)
                if male_index!=female_index:
                        #父母代个体
                        male=parents[male_index]
                        female=parents[female_index]
                        #交叉规则
                        left=random.randint(0,len(male)-3)
                        right=random.randint(left+1,len(male))
                        gene1=male[left:right]
                        gene2=female.copy()
                        for j in gene1:
                                gene2.remove(j)
                        child=gene1+gene2
                        # gene2=female[left:right]
                        #
                        # child1_c=male[right:]+male[:right]
                        # child2_c = female[right:] +female[:right]
                        # child1=child1_c.copy()
                        # child2 = child2_c.copy()
                        # for i in gene2:
                        #         child1_c.remove(i)
                        # for i in gene1:
                        #         child2_c.remove(i)
                        # child1[left:right]=gene2
                        # child2[left:right] = gene1
                        # child1[right:]=child1_c[0:len(child1)-right]
                        # child1[:left]=child1_c[len(child1)-right:]
                        #
                        # child2[right:] = child2_c[0:len(child1) - right]
                        # child2[:left] = child2_c[len(child1) - right:]
                        # children.append(child1)
                        # children.append(child2)
                        children.append(child)
        # if target_count>0:
        #         children=children[: target_count + 1]
        return children

#变异
def mutation(children):
        for i in range(len(children)):
                if random.random()<mutation_rate:
                        child=children[i]
                        u=random.randint(1,len(child)-4)
                        v=random.randint(u+1,len(child)-3)
                        w = random.randint(v+1, len(child) - 2)
                        child=child[0:u]+child[v:w]+child[u:v]+child[w:]
                        children[i]=child
                return children

#获得最好结果
def get_result(population):
        graded=[[get_total_distance(x),x] for x in population]
        graded=sorted(graded)
        return graded[0][0],graded[0][1]

index = [i for i in range(city_count)]
index.remove(origin)
population=[]
for i in range(count):
        population.append(index)
        #population.append(improve(index))

i=0
distance_list=[]
result_path_list=[]
while i<item_time:
        parents= selection(population)
        if len(parents)<count:
                children=crossover(parents)
                children=mutation(children)
                population=parents+children
        distance,result_path=get_result(population)
        distance_list.append(distance)
        result_path_list.append(result_path)
        i=i+1

def draw(bestPath):
        ax = plt.subplot(111, aspect='equal')
        ax.plot(cities[:, 0], cities[:, 1], 'x', color='blue')
        for i, city in enumerate(cities):
                ax.text(city[0], city[1], str(i))
        ax.plot(cities[bestPath, 0], cities[bestPath, 1], color='red')
        plt.show()

bestPath=[0]+result_path+[0]
draw(bestPath)
plt.plot(np.linspace(0,len(distance_list)-1,item_time),distance_list)
plt.show()

print("最短路径:%f"%distance_list[-1])
for i in bestPath:
        print(i,end='-->')
print("\t")
end = time.time()
print("运行时间:%.2f秒"%(end-start))

