# 文件功能： 实现 K-Means 算法

import numpy as np
import random
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.01, max_iter=500):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.resultdic = {}

    def distance(self, v1, v2):
        return np.sqrt(np.sum(np.square(v1 - v2)))
        
    def fit(self, data):
        # 作业1
        # 屏蔽开始
        currentJ = 0
        lastJ = 0
        centers=[]              
        disforJ=[]
        dis=[]
        iter_times=0
        result_current={}
        result_last={}
        same = False
        toler_count = 0
        centers=random.sample(list(data), self.k_) 
        while ((iter_times < self.max_iter_)):
       # and (toler_count <2)
        ######### update elements of every center #####
            result_current.clear() 
            disforJ.clear()            
            for x in data:
                for center in centers:           
                   dis.append(self.distance(x, center))
                center_index=dis.index(min(dis)) 
                if center_index in result_current.keys():               
                    result_current[center_index].append(x) 
                else:
                    result_current[center_index]=[]
                    result_current[center_index].append(x)              
                disforJ.append(min(dis))
                dis.clear()          
#            print(result_current)
#            print(disforJ)
        ########### update center ###############
            j=0
            for center in centers:
#                print('center', j, ':', center)
                center=np.array(result_current[j]).mean(axis=0)
#                print('newcenter:',center)
                j=j+1
            centers=list(centers)
#            print('newcenters',centers)
        #############calculate stop conditions#################
        
            iter_times = iter_times + 1    #iterate times
#            print('iter_times:',iter_times)
            
            currentJ = sum(disforJ)       # increment of J 
            toler_current = abs(currentJ-lastJ)
            if (toler_current < self.tolerance_ and toler_last < self.tolerance_):
                toler_count = toler_count + 1
            else:
                toler_count=0
            toler_last = toler_current
            
            same = (result_current == result_last) # elements of classes are not changed
            result_last = result_current
            
        self.resultdic = result_current
#        return result_current
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for x in p_datas:
            for key, values in self.resultdic.items():
                for value in values:
                    if all(value==x):
                        label = key
            result.append(label)                                               
        # 屏蔽结束
        return  result

if __name__ == '__main__':

    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
#    np.random.seed(42)   
#    x = np.random.randn(1000,2)
    
    k_means = K_Means(n_clusters=2)

    k_means.fit(x)

    cat = k_means.predict(x)

#    print(cat)
