import pandas as pd
import argparse
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries  

def load_data():
    pointWithNormal=pd.read_csv("airplane_0001.txt", header=None)
    pointWithNormal.columns=['x', 'y', 'z', 'nx', 'ny', 'nz']
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pointWithNormal[['x', 'y', 'z']].values)
    pcd.normals=o3d.utility.Vector3dVector(pointWithNormal[['nx', 'ny', 'nz']].values)
    return pcd
class issDetectors(object):
    def __init__(self,radius,gamma_21,gamma_32,nmsinterval):
        self.radius=radius    #用于计算某个点在该半径范围内的近邻点个数
        self.gamma_21=gamma_21
        self.gamma_32=gamma_32
        self.point_cloud_o3d = load_data()
        self.points=np.array(self.point_cloud_o3d.points)
        self.weight=[]  #每个点的权重
        self.pcd_tree=o3d.geometry.KDTreeFlann(self.point_cloud_o3d) #构建点云的kdtree，便于搜索近邻点     
        self.keypoint=[]     #未经过非极大值抑制的关键点的索引列表
        self.nmsinterval=nmsinterval#非极大值抑制获取最大特征值的间隔
        o3d.visualization.draw_geometries([self.point_cloud_o3d])
    def weightmatrix(self): #计算每个点的权重    
        for point in self.points:
            [_, idxs, _]=self.pcd_tree.search_radius_vector_3d(np.expand_dims(point,-1),self.radius) #搜索radius范围内的近邻点
#            print('idx',idxs)
            if(len(idxs)>1):
                res=1/len(idxs)   #取近邻点个数的倒数作为点的权重
            else:
                res=1
            self.weight.append(res) #更新每个点的权重           
                
    def caleig(self,query_point):
        localweight=[]
        [_, idxs, _] = self.pcd_tree.search_knn_vector_3d(np.expand_dims(query_point,-1),20)#搜索局部近邻点计算特征值
        near_points = self.points[idxs]
#        print('near_points',near_points)
        for idx in idxs:
           localweight.append(self.weight[idx])                 
        localweight=np.tile(np.array(localweight),(3,1))  
#        print(weight)
        near_points=np.array(near_points)
        near_points=near_points*localweight.T      #每个点赋权重
#        print('shape[0]',near_points.shape[0])
        return self.PCA(near_points)   #对加权的局部点云通过pca求特征值
        
    def PCA(self,data):    #PCA
        average= np.mean(data,axis=0)
        m, n = np.shape(data)
        data_adjust = []
        avgs = np.tile(average, (m, 1))##沿着列方向把average复制m次
        data_adjust = data - avgs
        covX = np.cov(data_adjust.T)   #计算协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(covX) 
        sort = eigenvalues.argsort()[::-1]##从大到小排列
        eigenvalues = eigenvalues[sort]
        return eigenvalues
    def keypoints(self):#获取关键点
        i=0
        lambda3=[]
        for point in iter(self.points):
#            print(i)
            eig=self.caleig(point)  #求该点在局部范围内的特征值         
            if (not(abs(eig[0]-eig[1])<0.000000003) and not(abs(eig[1]-eig[2])<0.000000003) and (eig[1]/(eig[0]+0.0000001))<self.gamma_21 and (eig[2]/(eig[1]+0.0000001))<self.gamma_32):#作为关键点的条件
                self.keypoint.append(i)
                lambda3.append(eig[2])                 
#                print('eigenvalues',eig)
            i=i+1
#            print(self.keypoint)
        print('num of keypoint:',len(self.keypoint))
        return self.keypoint,lambda3
    
    def NMS(self):
        keypointindex=[]
        idxs,lambda3=self.keypoints()
        i=0
        while(i<len(idxs)):
            keyindex=lambda3.index(max(lambda3[i:i+self.nmsinterval]))#固定间隔内获取最大值的索引
            keypointindex.append(idxs[keyindex])
            i=i+self.nmsinterval
        return keypointindex
    
        #返回idx序列
def open3dplot(points,keypoints_idx):#使用open3d显示点云及点云的关键点
    colors=np.zeros(shape=(len(points),3))  ##每个点都需要一个颜色向量
    for i in keypoints_idx:
        colors[i]=np.array([255,0,0]) #关键点为红色        
    pointCloud=PointCloud()
    pointCloud.points=Vector3dVector(points)  
    pointCloud.colors=Vector3dVector(colors)
    draw_geometries([pointCloud])    
    
def main():
    dector=issDetectors(0.3,0.1,0.1,10)
    dector.weightmatrix()
    keypoints_idx,_=dector.keypoints()
#    keypoints_idx=dector.NMS()
    open3dplot(dector.points,keypoints_idx)
if __name__ == '__main__':
    main()   

