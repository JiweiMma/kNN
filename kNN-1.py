from numpy import shape, tile
from numpy.ma import zeros
import operator

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

#数据的可视化

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',markersize=6, label='1')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='2')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='3')
    # 添加进图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()

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

#验证分类器的正确率，本次数据没有排序，所以可以随机选择10%的数据进行测试
def datingClassTest():
    hoRatio=0.10#选择10%的测试数据
    filename="DataTestSet2.txt"
    datingMat,datingLabels=filematrix(filename)#将返回的特征矩阵和分类向量分别存储
    normMat,ranges,minVals=autoNorm(datingDataMat)#归一化特征值
    m=normMat.shape[0]#获取行数,本次为1000
    numTestVecs=int(m*hoRatio)#得到10%的测试数据的个数
    errorCount=0.0#分类错误的计数，初始化为0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs作为训练集
        classiferResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("分类结果：%d\t真实类别：%d" %(classiferResult,datingLabels[i]))
        if(classiferResult !=datingLabels[i]):
            errorCount+=1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))





if __name__=='__main__':
    filename="DataTestSet2.txt"
    datingDataMat,datingLabels=filematrix(filename)
    normDataSet,ranges,minVals=autoNorm(datingDataMat)
    print(datingDataMat)
    print(datingLabels)
    showdatas(datingDataMat,datingLabels)
    print(normDataSet)
    print(ranges)
    print(minVals)
    datingClassTest()
