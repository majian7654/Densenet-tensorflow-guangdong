import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
正常	norm
不导电	defect1
擦花	defect2
横条压凹	defect3
桔皮	defect4
漏底	defect5
碰伤	defect6
起坑	defect7
凸粉	defect8
涂层开裂	defect9
脏点	defect10
其他	defect11
"""

label_warp = {'norm': 0,
              'defect1': 1,
              'defect2': 2,
              'defect3': 3,
              'defect4': 4,
              'defect5': 5,
              'defect6': 6,
              'defect7': 7,
              'defect8': 8,
              'defect9': 9,
              'defect10': 10,
              'defect11': 11,
              }



class lvData:
    def __init__(self, trainDataPath = './data/label.csv'):
        self.trainDataPath = trainDataPath
        self.fig = plt.figure()
        self.data = pd.read_csv(trainDataPath)
    
    def prepare_data(self):#split train and val data
        # 分离训练集和测试集，stratify参数用于分层抽样
        train_x, val_x, train_y, val_y = train_test_split(self.data['img_path'], self.data['label'], test_size=0.3, random_state=666, stratify=self.data['label'])
        return train_x, train_y, val_x, val_y 

    def drawBar(self):
        label = self.data['label']
        label_count = label.value_counts()
        x = ['norm','defect1','defect2','defect3','defect4','defect5','defect6','defect7','defect8','defect9','defect10','defect11']
        y = [label_count[label_warp[_]] for _ in x]
        print(y)
        ind = np.arange(len(x))    # the x locations for the groups
        ax = self.fig.add_subplot(111)
        ax.bar(ind, y)
        plt.xticks(ind, x)
        plt.show()

    def drawPie(self):
        label = self.data['label']
        label_count = label.value_counts()
        x = ['norm','defect1','defect2','defect3','defect4','defect5','defect6','defect7','defect8','defect9','defect10','defect11']
        y = [label_count[label_warp[_]] for _ in x]
        print(y)
        ax = self.fig.add_subplot(111)
        explode = (0.1, 0, 0, 0,0,0,0,0,0,0,0,0)
        ax.pie(y, explode=explode, labels=x, 
        autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()


if __name__=='__main__':
    #dalz = dataAnalyze()
    #dalz.drawPie()
    #dalz.drawBar()
    data = lvData()
    train_x, train_y, val_x, val_y = data.prepare_data()
    print(val_x)
#    data.drawPie()
