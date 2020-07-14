import numpy as np  #导入计算包numpy
import operator  #导入运算符模块

def createDataSet():  #创建训练样本集和对应的标签向量
    group=np.array([[1.0,1.1],[1.0,1.0],[0,1],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):  #inX输入向量，dataSet是训练样本集矩阵，labels是训练样本的标签向量，K分类的值
    dataSetSize=dataSet.shape[0] #训练样本的个数
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDist=(diffMat**2).sum(axis=1)  #1表述行，0表述列
    dist=sqDist**0.5  #得到输入向量与各训练样本的欧式距离
    sortedIndicies=dist.argsort() #距离从小到大排序，返回对应的列表索引值
    classCount={} #eg :  classCount{A:2}
    for i in range(k):
        label=labels[sortedIndicies[i]]  #获得类别标签
    classCount[label]=classCount.get(label,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #降序排列
    return sortedClassCount[0][0]

if __name__=='__main__':
    group,labels=createDataSet()
print(classify0([0.0],group,labels,3))