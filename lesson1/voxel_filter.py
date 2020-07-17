# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸

def voxel_filter(point_cloud, leaf_size):

    filtered_points = []
    # 作业3
    # 屏蔽开始

    r=leaf_size
    
    point_cloud=np.array(point_cloud)
    
    Dx=(np.max(point_cloud[:,0])-np.min(point_cloud[:,0]))/r
    
    Dy=(np.max(point_cloud[:,1])-np.min(point_cloud[:,1]))/r
    
    Dz=(np.max(point_cloud[:,2])-np.min(point_cloud[:,2]))/r
    
    dic={}
    
    for point in point_cloud:
    
        hx=np.floor((point[0]-np.min(point_cloud[:,0]))%r)
        
        hy=np.floor((point[1]-np.min(point_cloud[:,1]))%r)
        
        hz=np.floor((point[2]-np.min(point_cloud[:,2]))%r)
        
        h=hx+hy*Dx+hz*Dx*Dy
        
        if h in dic.keys():
           
           dic[h].append(point)

        else:
         
           dic[h]=[]

           dic[h].append(point)
       
    for i in dic.keys():
    
        filtered_points.append(np.median(dic[i],axis=0))

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/xcy/myWork/pointCloudProcessing/ModelNet40/ply_data_points/airplane/train/airplane_0001.ply"
    
    point_cloud_pynt = PyntCloud.from_file(file_name)

    print(point_cloud_pynt.points.shape)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    
     
#    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    
    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100)
    
    print(filtered_cloud.shape)
    
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
    
if __name__ == '__main__':
    main()
