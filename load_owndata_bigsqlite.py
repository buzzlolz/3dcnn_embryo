import os
import cv2
import numpy as np
import time
import copy

def get_total_datanum():
    num=0

    for i in range(1,9):
    
        root_path = '/home/n200/D-slot/3dcnn_sqlitedata_1119/'+str(i)+'/FP00'

        data_list = os.listdir(root_path)
        print('len datalist:',i,len(data_list))
        num+=len(data_list)
    return num

def get_min_datanum():
    list_savenum=[]
    for i in range(1,9):
    
        root_path = '/home/n200/D-slot/3dcnn_sqlitedata_1119/'+str(i)+'/FP00'
        file_num=len(os.listdir(root_path))
        list_savenum.append(file_num)

    num_min = int(min(list_savenum)/7)

    print(num_min)

    return num_min


def load_own_data(label,each_stage_get_num):

    root_path = '/home/n200/D-slot/3dcnn_sqlitedata_1119/'+str(label)
    img_folder = '/home/n200/D-slot/3dcnn_sqlitedata_1119/'+str(label)+'/FP00'
    file_num=len(os.listdir(img_folder))
    test=np.empty(shape=[each_stage_get_num,128, 128,7])
    interval = int(file_num/each_stage_get_num)
    print("interval",interval)
    print('file_num',file_num)
    total_img=[]
    count=0

    each_fp_filename_list = []

    for i in range(7):
        fn = 'FP0'+str(i)
        fd_path = os.path.join(root_path,fn)
        file_list = os.listdir(fd_path)
        each_fp_filename_list.append(file_list)
    print(np.array(each_fp_filename_list).shape)
    # print(each_fp_filename_list[1])

    for num_img in range(len(os.listdir(img_folder))):
        

        t1=time.time()
        # print(num_img)

        if num_img%interval==0 and num_img!=0 and count<each_stage_get_num:

            image_array = np.empty(shape=(128,128,7))

            for i in range(7):
                fn = 'FP0'+str(i)
                fd_path = os.path.join(root_path,fn)
                # print(fn)
                # print(sorted(os.listdir(fd_path))[num_img])
                
                # filename=sorted(os.listdir(fd_path))[num_img]
                filename=each_fp_filename_list[i][num_img]
                filename_path = os.path.join(fd_path,filename)
                
                
                image = cv2.imread(filename_path)
                image= cv2.resize(image,(128,128))
                image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                # print(np.array(image).shape) 
                # cv2.imshow('test',image)
                image = np.array(image,np.float32)
                # image = np.expand_dims(image, axis=2)
                # print(image_array[:,:,0].shape)
                # print(image.shape)
                image_array[:,:,i]=copy.copy(image)
                # if i ==0:
                #     image_array=image
                # else:
                #     image_array=np.concatenate([image_array,image], axis=2)

            
            print('num_img',num_img)
            print("count:",count)
            test[count]=copy.copy(image_array)
            count+=1
            t2=time.time()
            print('spend time:',t2-t1)
        # print(num_img)

        # print(test.shape)

            
                # print('image array shape',image_array.shape,filename_path)
            # print(image_array.shape)

            # if cv2.waitKey(0) & 0xFF==ord('q'):
            #     break
        
        

        
        # if num_img==0:
        #     image_array = np.expand_dims(image_array, axis=0)
        #     total_img=  image_array
        # elif num_img%8==0:
        #     image_array = np.expand_dims(image_array, axis=0)
        #     total_img=np.concatenate([total_img,image_array], axis=0)
        # # print(total_img.shape)

        #     # cv2.waitKey(0)
        #     # for filename in sorted(os.listdir(fd_path)):
        #     #     print(fn)
        #     #     filename_path = os.path.join(fd_path,filename)
        #     #     print(filename_path[0])
        # print("total img shape:",np.array(total_img).shape)
    # print(test)
    print("each_stage_get_num",each_stage_get_num)
    print('len stage list:',len(test))

    return len(test), test



if __name__ == "__main__":
    # get_total_datanum()
    get_min_datanum()