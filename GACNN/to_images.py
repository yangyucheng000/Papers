
from PIL import Image
import numpy as np
import pickle
from tqdm import trange
from os.path import join
import os, random, shutil


def my_mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    return dict

def deldir(dir):
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            deldir(t)#重新调用次方法
        else:
            os.unlink(t)
    os.rmdir(dir)#递归删除目录下面的空文件夹



# settings
src_dir = './cifar-100-python' # the dir you uncompress the dataset
dst_dir = './datasets1/' # the dir you want the img_dataset to be


if __name__ == '__main__':

    deldir('./datasets1')
    meta = unpickle(join(src_dir, 'meta')) # KEYS: {'fine_label_names', 'coarse_label_names'}
    my_mkdirs(dst_dir)

    for data_set in ['train', 'test']:

        print('Unpickling {} dataset......'.format(data_set))
        data_dict = unpickle(join(src_dir, data_set)) # KEYS: {'filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'}
        #my_mkdirs(join(dst_dir, data_set))

        for fine_label_name in meta['fine_label_names']:
            #my_mkdirs(join(dst_dir, data_set, fine_label_name))
            if data_set=='train':
                path='./datasets1/train_'+fine_label_name+'/1'
                my_mkdirs(path)
            else:
                path='./datasets1/test_0/'+fine_label_name
                my_mkdirs(path)



        for i in trange(data_dict['data'].shape[0]):
            img = np.reshape(data_dict['data'][i], (3, 32, 32))
            i0 = Image.fromarray(img[0])
            i1 = Image.fromarray(img[1])
            i2 = Image.fromarray(img[2])
            img = Image.merge('RGB', (i0, i1, i2))
            #print(meta['fine_label_names'][data_dict['fine_labels'][i]])      20个类别名称
            if data_set=='train':
                path='./datasets1/train_'+meta['fine_label_names'][data_dict['fine_labels'][i]]+'/1'
                #img.save(join(path, data_dict['filenames'][i]))
            else:
                path='./datasets1/test_0/'+meta['fine_label_names'][data_dict['fine_labels'][i]]
            img.save(join(path, data_dict['filenames'][i]))


    #输出所有文件夹
    for dirpath, dirnames, filenames in os.walk('./datasets1'):
        for dirname in dirnames:
            if dirname=='1' and dirpath!='./datasets1/test_0':
                #print(os.path.join(dirpath, dirname))
                my_mkdirs(dirpath + '/0')
                tarDir=dirpath+'/0'


                for dirpath1, dirnames1, filenames1 in os.walk('./datasets1'):
                    for dirname1 in dirnames1:
                        if dirname1 == '1' and dirpath1 != './datasets1/test_0' and dirpath1!=dirpath:

                            #print(os.path.join(dirpath1, dirname1))
                            pathDir = os.listdir(os.path.join(dirpath1, dirname))  # 取图片的原始路径
                            filenumber = len(pathDir)
                            #print(filenumber)
                            rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
                            picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
                            #print(picknumber)
                            sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
                            #print(sample)
                            for name in sample:
                                shutil.copy(os.path.join(dirpath1, dirname1) +'/'+ name, tarDir +'/'+ name)






    print(len(os.listdir('./datasets1/train_plain/0')))
    print(len(os.listdir('./datasets1/train_raccoon/1')))
    print(len(os.listdir('./datasets1/test_0/boy')))




    print('All done.')

