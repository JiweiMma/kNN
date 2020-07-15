from Tools.scripts.treesync import raw_input
from numpy import shape, tile
from numpy.ma import zeros
import operator
import numpy as np

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
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #降序排列  reverse
    return sortedClassCount[0][0]

#将数据的格式，改变为分类器可以接受的格式。
#输入为文件名字符串，输出为训练样本矩阵和类标签向量。
def filematrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()  #得到了列表
    numberOfLines=len(arrayOLines)#返回字符串，列表，字典，元组等长度
    returnMat=zeros((numberOfLines,3)) #创建为0的数组
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()#用于移除字符串头尾指定的字符
        listFromLine=line.split('\t')#将整行数据分割成元素列表
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

#归一化特征值，处理不同取值范围的特征值，将范围值处理为0到1或者-1到1之间
def autoNorm(dataSet):
    minVals=dataSet.min(0) #取每列的最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))#待归一化的新特征矩阵
    m=dataSet.shape[0]#样本总数
    normDataSet=dataSet-tile(minVals,(m,1))#tile函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#预测函数
def classifyPerson():
    #输出结果
    resultList=['不喜欢','一般喜欢','特别喜欢']
    #特征用户的输入
    precentTats=float(raw_input("玩视频游戏所耗时间百分比:"))
    ffMiles=float(raw_input("每年获得的飞行常客里程数:"))
    iceCream=float(raw_input("每周消费的冰淇淋公升数:"))
    filename="DataTestSet2.txt"
    #打开并处理数据
    datingDataMat,datingLabels=filematrix(filename)
    #训练集的归一化
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #生成测试集
    inArr=np.array([ffMiles,precentTats,iceCream])
    #把测试集归一化
    norminArr=(inArr-minVals)/ranges
    #返回分类结果
    classifierResult=classify0(norminArr,normMat,datingLabels,3)
    #打印输出结果
    print("海伦可能%s这个人"%(resultList[classifierResult-1]))

if __name__=='__main__':
    classifyPerson()
