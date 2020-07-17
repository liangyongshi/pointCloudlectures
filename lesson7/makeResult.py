import os
import random
pwd_dir = os.getcwd()  #返回文件所在的当前目录
path = pwd_dir + '/label_1'

file_list = os.listdir(path) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
#print(file_list)
num = len(file_list)

for i in range(num):

    fr = open(path+'/'+file_list[i])
    with open(pwd_dir + '/final_result/' + file_list[i],"w") as fw:
        lines = fr.readlines()
        for j in range(len(lines)):
            if lines[j][0:10:1]=='Pedestrian' or lines[j][0:3:1]=='Car' or lines[j][0:7:1] == 'Cyclist' or lines[j][0:8:1]=='DontCare':
              
                lines[j] = lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n' + lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n' + lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n'
                fw.write(lines[j])
                print(lines[j])
            else:
                print(lines[j])

                local = lines[j].find(' ')
                lines[j] = lines[j][local+1:]
                alter = ['Pedestrian','Car','Cyclist']
                index = random.sample(range(0,3),1)[0]
                head = alter[index]
                lines[j] = head + ' ' + lines[j]
                lines[j] = lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n' + lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n' + lines[j].replace('\n',' ') + ' ' + str(random.random())+'\n'
                fw.write(lines[j])


