'''
本文为机器学习实战的例子
理解，复述，练习，整理
鼓励自己，坚持，哈哈哈
'''

#第一个例子
#第一步：import module，如numpy,operator
#第二步：提供（创建）已分类的数据集，标签向量组
#第三步：创建分类器
#                     步骤：1，获取已分类数据集的维数
#                               2，未分类数据根据获取的维数生成同样维数的数据集
#                               3，求两组数据集的欧氏距离，进行升序排序并获取索引值
#                               4，获取前k个索引值对应的标签，
#                               5，统计标签出现的次数，以标签为字典的键，次数为值
#                               5，对字典内容进行降序排序（根据值），值最多的项对应的标签即分类结果
#from numpy import *  #不使用这个语句，试试哪些函数需要用numpy.
import numpy as np
import operator

def createDataSet():
    group = np.array([[4.1,1.1],[2.1,1.2],[0.4,5.2],[2.2,3.1]])
    labels = ['A','B','A','B']
    return group,labels

def classify1(inX,group,labels,k):
    dRow = group.shape[0]#获取行数
    dataset = np.tile(inX,(dRow,1)) - group#得到的形式[[x,y],[x,y],[x,y],[x,y]]-group=[[a,b],[c,d],[e,f],[g,h]]
    distances2 = dataset**2#[[a^2,b^2],[c^2,d^2],...]
    distances = np.sum(distances2,axis=1)#[a^2+b^2,c^2+d^2,...]
    Odistances =distances**0.5#[ab,cd,ef,gh]
    sortedindex =Odistances.argsort()#得到排序后的索引，如[2,1,0,3],对应ef<cd<ab<gh
    classCount={}#创建空字典
    for i in range(k):
        Vlabels = labels[sortedindex[i]]#sortedindex[0]=2,labels[2]='B'，最终得到['A','B','A']
        classCount[Vlabels] = classCount.get(Vlabels,0)+1#classCount[Vlabels]:以Vlabels为键值
                                                                                    #.get(key,default),返回key对应的值
                                                                                    #{'A':2,'B':1}
    #{}.items(),以列表形式返回[(key,value),(key,value),...]
    #operator.itemgetter(0)：按键排序,operator.itemgetter(1)：按值排序
    #reverse=True，降序
    sortedCount = sorted(classCount.items(),
                               key=operator.itemgetter(1),reverse=True)
    print(sortedCount)
    return sortedCount[0][0]
group,labels=createDataSet()
print(classify1([4,6],group,labels,4))

