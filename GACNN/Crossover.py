import random
import numpy as np
import copy
from update_dict import update_dict
from utils import StatusUpdateTool, Utils
import populations


keys_layers=[[],[],[],[],[],[],[],[]]#保存每一层的节点，其中0和8层为空
key_list = []
def get_dict_allkeys(dict_a):
    """
    遍历嵌套字典，获取json返回结果的所有key值
    :param dict_a:
    :return: key_list
    """
    if isinstance(dict_a, dict):  # 使用isinstance检测数据类型
        # 如果为字典类型，则提取key存放到key_list中
       # print('1')
        for x in range(len(dict_a)):
            temp_key = list(dict_a.keys())[x]
            if temp_key[-1]=='1':
                keys_layers[1].append(temp_key)
            elif temp_key[-1]=='2':
                keys_layers[2].append(temp_key)
            elif temp_key[-1] == '3':
                keys_layers[3].append(temp_key)
            elif temp_key[-1] == '4':
                keys_layers[4].append(temp_key)
            elif temp_key[-1] == '5':
                keys_layers[5].append(temp_key)
            elif temp_key[-1] == '6':
                keys_layers[6].append(temp_key)
            elif temp_key[-1] == '7':
                keys_layers[7].append(temp_key)

            #print(temp_key)
            temp_value = dict_a[temp_key]
            key_list.append(temp_key)
            get_dict_allkeys(temp_value)  # 自我调用实现无限遍历
    elif isinstance(dict_a, list):
        # 如果为列表类型，则遍历列表里的元素，将字典类型的按照上面的方法提取key
        for k in dict_a:
            if isinstance(k, dict):
                for x in range(len(k)):
                    temp_key = list(k.keys())[x]

                    temp_value = k[temp_key]
                    key_list.append(temp_key)
                    get_dict_allkeys(temp_value)  # 自我调用实现无限遍历
    return key_list


def get_dict_vaules(dict_a,key, default=None):
    """
    遍历嵌套字典，获取json返回结果的所有key值
    :param dict_a:
    :return: key_list
    """
    tmp = dict_a

    for k, v in tmp.items():

        if k == key:

            return v

        else:

            if isinstance(v, dict):

                ret = get_dict_vaules(v, key, default)

                if ret is not default:

                    return ret

    return default

def update(key,dict_data):
    #判断需要修改的key是否在初始字典中，在则修改
    if key in dict_data:
        #将key为'gg'的值修改成'张三'
        dict_data[key]='张三'
        #print(dict_data)
        #循环字典获取到所有的key值和value值
        for keys,values in dict_data.items():
            #判断valus值是否为列表或者元祖
            if isinstance(values,(list,tuple)):
                #循环获取列表中的值
                for i in values:
                    #判断需要修改的值是否在上一个循环出的结果中，在则修改
                    if key in i and isinstance(i,dict):
                        #调用自身修改函数，将key的值修改成'张三'
                        update(key,i)
                    else:
                        #否者则调用获取value函数
                        get_value(i)
            elif isinstance(values,dict):
                if key in values:
                    update(key,values)
                else:
                    for keys,values in values.items():
                        if isinstance(values,dict):
                            update(key, values)
    else:
        #循环获取原始字典的values值
        for keys,values in dict_data.items():
            #判断values值是否是列表或者元祖
            if isinstance(values,(list,tuple)):
                #如果是列表或者元祖则循环获取里面的元素值
                for i in values:
                    #判断需要修改的key是否在元素中
                    if key in i:
                        #调用修改函数修改key的值
                        update(key,i)
                    else:
                        #否则调用获取values的值函数
                        get_value(i)
            #判断values值是否为字典
            elif isinstance(values,dict):
                #判断需要修改的key是否在values中
                if key in values:
                    #调用修改函数修改key的值
                    update(key,values)
                else:
                    #获取values值的函数
                    get_value(values)
    return dict_data


def get_value(tt):
    #循环获取values的值
    for values in tt.values():
        #判断循环出的value的值是否为列表或者元祖
        if isinstance(values, (list,tuple)):
            #如果为列表或者元祖则循环获取列表或者元祖中的值
            for i in values:
                #判断需要修改的值是否在循环出的值中,且i为字典
                if key in i and isinstance(i,dict):
                    #调用修改函数，将key的值修改为'张三'
                    update(key, i)
                else:
                    #否则调用获取value函数
                    get_value(i)
        elif isinstance(values,dict):
            if key in values:
                update(key, values)
            else:
                get_value(values)



class CrossoverAndMutation(object):
    def __init__(self,prob_crossover, prob_mutation, individuals, _params=None):
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.individuals = individuals
        self.params = _params # storing other parameters if needed, such as the index for SXB and polynomial mutation
        #self.log = _log
        self.offspring = []


    def process(self,curgen):
        mutation = Mutation(self.individuals, self.prob_mutation)
        offspring =mutation.do_mutation()
        self.individuals = offspring
        #Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])

        crossover = Crossover(self.individuals, self.prob_crossover)
        offspring = crossover.do_crossover(curgen)
        self.offspring = offspring
        #print(self.offspring)
        #Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])

        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d'%(self.params['gen_no'], i)
            indi.indi_no = indi_no
        return offspring

    def individuals_to_string(self):
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)



class Crossover(object):
    def __init__(self, individuals, prob_):
        self.individuals = individuals
        self.prob = prob_
        #self.key_list = []
        # self.log = _log
        # self.pool_limit = StatusUpdateTool.get_pool_limit()[1]


    """
    binary tournament selection
    """
    def _choose_two_diff_parents(self,k):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random() * count_))
        idx2 = int(np.floor(np.random.random() * count_))

        while idx2 == idx1 or idx1==k or idx2==k:
            idx1 = int(np.floor(np.random.random() * count_))
            idx2 = int(np.floor(np.random.random() * count_))

        return idx1, idx2



    def do_crossover(self,curgen):

        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        """
            选择准确率最大的一个个体
        """
        k = 0
        maxacc = -2
        for i in range(len(self.individuals)):
            if self.individuals[i].acc > maxacc:
                maxacc = self.individuals[i].acc
                k = i

        for _ in range(len(self.individuals) // 3):

            ind2 = random.randint(0, len(self.individuals) - 1)
            while ind2 == k:  #确保父母不同
                ind2 = random.randint(0, len(self.individuals) - 1)
            parent1 = copy.deepcopy(self.individuals[k])
            parent2 = copy.deepcopy(self.individuals[ind2])

            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
                """
                交换子树
                """
                keys_layers[0].clear()
                keys_layers[1].clear()
                keys_layers[2].clear()
                keys_layers[3].clear()
                keys_layers[4].clear()
                keys_layers[5].clear()
                keys_layers[6].clear()
                keys_layers[7].clear()
                key_list.clear()
                parent2_keys = get_dict_allkeys(parent2.tree_dict)  #得到parent2所有节点
                if curgen<15:
                    root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  #随机选一层
                                                  p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0])
                else:
                    root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                  p=[0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0])


                root2 = random.choice(keys_layers[int(root_layer)])#随机选一个根
                subtree2 = get_dict_vaules(parent2.tree_dict, root2)#该根的子树
                layer = root2[-1]
                keys_layers[0].clear()
                keys_layers[1].clear()
                keys_layers[2].clear()
                keys_layers[3].clear()
                keys_layers[4].clear()
                keys_layers[5].clear()
                keys_layers[6].clear()
                keys_layers[7].clear()
                key_list.clear()
                parent1_keys = get_dict_allkeys(parent1.tree_dict) #得到parent1所有节点
                root1 = random.choice(keys_layers[int(layer)])
                subtree1 = get_dict_vaules(parent1.tree_dict, root1)

                tree1 = {}#进行交换
                tree2 = {}
                tree1[root2] = subtree1#parent2的根+parent1的子树
                tree2[root1] = subtree2#parent1的根+parent2的子树
                #print('tree1:',tree1)
                #print('tree2:',tree2)

                parent1.tree_dict = update_dict(parent1.tree_dict, tree2)
                parent2.tree_dict = update_dict(parent2.tree_dict, tree1)

                while parent1.tree_dict == self.individuals[k].tree_dict or parent2.tree_dict == self.individuals[k].tree_dict :#如果交换后的
                    keys_layers[0].clear()
                    keys_layers[1].clear()
                    keys_layers[2].clear()
                    keys_layers[3].clear()
                    keys_layers[4].clear()
                    keys_layers[5].clear()
                    keys_layers[6].clear()
                    keys_layers[7].clear()
                    key_list.clear()
                    parent2_keys = get_dict_allkeys(parent2.tree_dict)
                    if curgen < 15:
                        root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                      p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0])
                    else:
                        root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                      p=[0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0])

                    root2 = random.choice(keys_layers[int(root_layer)])

                    subtree2 = get_dict_vaules(parent2.tree_dict, root2)
                    layer = root2[-1]
                    keys_layers[0].clear()
                    keys_layers[1].clear()
                    keys_layers[2].clear()
                    keys_layers[3].clear()
                    keys_layers[4].clear()
                    keys_layers[5].clear()
                    keys_layers[6].clear()
                    keys_layers[7].clear()
                    key_list.clear()
                    parent1_keys = get_dict_allkeys(parent1.tree_dict)
                    root1 = random.choice(keys_layers[int(layer)])
                    subtree1 = get_dict_vaules(parent1.tree_dict, root1)

                    tree1 = {}
                    tree2 = {}
                    tree1[root2] = subtree1
                    tree2[root1] = subtree2
                    #print(tree1)
                    #print(tree2)

                    parent1.tree_dict = update_dict(parent1.tree_dict, tree2)
                    parent2.tree_dict = update_dict(parent2.tree_dict, tree1)

                parent1.l12_list.clear()
                parent2.l12_list.clear()
                parent1.paths = parent1.dg(parent1.tree_dict)
                parent2.paths = parent2.dg(parent2.tree_dict)
                offspring1, offspring2 = parent1, parent2
                # print('after-c:', offspring1.tree_dict)
                offspring1.reset_acc()
                offspring2.reset_acc()
                new_offspring_list.append(offspring1)
                new_offspring_list.append(offspring2)
                # new_offspring_list.append(offspring2)
            else:
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)
        for _ in range(len(self.individuals) // 6):
            ind1, ind2 = self._choose_two_diff_parents(k)

            parent1, parent2 = copy.deepcopy(self.individuals[ind1]), copy.deepcopy(self.individuals[ind2])
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
                keys_layers[0].clear()
                keys_layers[1].clear()
                keys_layers[2].clear()
                keys_layers[3].clear()
                keys_layers[4].clear()
                keys_layers[5].clear()
                keys_layers[6].clear()
                keys_layers[7].clear()
                key_list.clear()
                parent2_keys = get_dict_allkeys(parent2.tree_dict)
                if curgen < 15:
                    root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                  p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0])
                else:
                    root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                  p=[0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0])

                root2 = random.choice(keys_layers[int(root_layer)])
                subtree2 = get_dict_vaules(parent2.tree_dict, root2)
                layer = root2[-1]
                keys_layers[0].clear()
                keys_layers[1].clear()
                keys_layers[2].clear()
                keys_layers[3].clear()
                keys_layers[4].clear()
                keys_layers[5].clear()
                keys_layers[6].clear()
                keys_layers[7].clear()
                key_list.clear()
                parent1_keys = get_dict_allkeys(parent1.tree_dict)
                root1 = random.choice(keys_layers[int(layer)])
                subtree1 = get_dict_vaules(parent1.tree_dict, root1)

                tree1 = {}
                tree2 = {}
                tree1[root2] = subtree1
                tree2[root1] = subtree2
                #print('tree1:',tree1)
                #print('tree2:',tree2)
                parent1.tree_dict = update_dict(parent1.tree_dict, tree2)
                parent2.tree_dict = update_dict(parent2.tree_dict, tree1)

                while parent1.tree_dict == self.individuals[k].tree_dict or parent2.tree_dict == self.individuals[k].tree_dict :
                    keys_layers[0].clear()
                    keys_layers[1].clear()
                    keys_layers[2].clear()
                    keys_layers[3].clear()
                    keys_layers[4].clear()
                    keys_layers[5].clear()
                    keys_layers[6].clear()
                    keys_layers[7].clear()
                    key_list.clear()
                    parent2_keys = get_dict_allkeys(parent2.tree_dict)  # 随机parent2的一个节点
                    if curgen < 15:
                        root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                      p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0])
                    else:
                        root_layer = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],  # 随机选一层
                                                      p=[0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0])

                    root2 = random.choice(keys_layers[int(root_layer) ])
                    subtree2 = get_dict_vaules(parent2.tree_dict, root2)

                    layer = root2[-1]
                    keys_layers[0].clear()
                    keys_layers[1].clear()
                    keys_layers[2].clear()
                    keys_layers[3].clear()
                    keys_layers[4].clear()
                    keys_layers[5].clear()
                    keys_layers[6].clear()
                    keys_layers[7].clear()
                    key_list.clear()
                    parent1_keys = get_dict_allkeys(parent1.tree_dict)
                    root1 = random.choice(keys_layers[int(layer)])

                    subtree1 = get_dict_vaules(parent1.tree_dict, root1)

                    tree1 = {}
                    tree2 = {}
                    tree1[root2] = subtree1
                    tree2[root1] = subtree2

                    parent1.tree_dict = update_dict(parent1.tree_dict, tree2)
                    parent2.tree_dict = update_dict(parent2.tree_dict, tree1)

                parent1.l12_list.clear()
                parent2.l12_list.clear()
                parent1.paths = parent1.dg(parent1.tree_dict)
                parent2.paths = parent2.dg(parent2.tree_dict)
                offspring1, offspring2 = parent1, parent2
                # print('after-c:', offspring1.tree_dict)
                offspring1.reset_acc()
                offspring2.reset_acc()
                new_offspring_list.append(offspring1)
                new_offspring_list.append(offspring2)
            else:
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        print('CROSSOVER-%d offspring are generated, new:%d, others:%d' % (
        len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list


class Mutation(object):

    def __init__(self, individuals, prob_):
        self.individuals = individuals
        self.prob = prob_

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random()*count_))
        #print(idx1)
        idx2 = int(np.floor(np.random.random()*count_))
        #print(idx2)
        while idx2 == idx1:
            idx2 = int(np.floor(np.random.random()*count_))

        if self.individuals[idx1].acc > self.individuals[idx1].acc:
            return idx1
        else:
            return idx2

    def do_mutation(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        for _ in range(len(self.individuals)):
            ind1 = self._choose_one_parent()

            parent1= copy.deepcopy(self.individuals[ind1])

            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_from_parent'] += 1
                #print('before-m:', parent1.tree_dict)
                print('before-m:', parent1.tree_dict)
                parent1.l12_list.clear()
                parent1.tree_dict=parent1.create_tree(100)#随机生成一个树
                parent1.paths = parent1.dg(parent1.tree_dict)
                print('after-m:', parent1.tree_dict)
                offspring1 = parent1
                new_offspring_list.append(offspring1)

            else:
                _stat_param['offspring_from_parent'] += 1
                new_offspring_list.append(parent1)


        print('mutation-%d offspring are generated, new:%d, others:%d' % (
        len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list





if __name__ == '__main__':
    #m = Mutation(None, None, None)
    #m.do_mutation()
    print(int('1'))

    tree1={'0-1': {'0-2': {'0-3': {'0-4': {'0-5': {'0-6': {'0-7': {'0-8': None}}}}}},
             '1-2': {'1-3': {'1-4': {'1-5': {'1-6': {'1-7': {'1-8': None}}}}}},
             '2-2': {'2-3': {'2-4': {'2-5': {'2-6': {'2-7': {'2-8': None}}}}}},
             '3-2': {'3-3': {'3-4': {'3-5': {'3-6': {'3-7': {'3-8': None}}}}}},
             '4-2': {'4-3': {'4-4': {'4-5': {'4-6': {'4-7': {'4-8': None}}}}}},
             '5-2': {'5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}}},
             '6-2': {'6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},
             '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},
             '8-2': {'8-3': {'8-4': {'8-5': {'8-6': {'8-7': {'8-8': None}}}}}},
             '9-2': {'9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}}}}

    subtree= {'1-2':"1"}

    #ans=update_dict(tree1, subtree)
    #print(ans)
    parent1_keys = get_dict_allkeys( tree1)
    # layer1=[1,2,3,4,5]
    # layer2 = [11, 21, 31, 41, 51]
    # layer3 = [111, 211, 311, 411, 511]
    layers=[]

    print(keys_layers)

    #root_layer = np.random.choice([1,2,3,4,5,6,7,8],p=[0.05,0.3,0.25,0.15,0.1,0.05,0.05,0.05])
    root_layer = np.random.choice([1, 2], p=[0.7, 0.3])
    root1=random.choice(keys_layers[root_layer])
    print(root1)
    subtree1 = get_dict_vaules(tree_dict, root1)
    print(subtree1)
    print(subtree1.keys()[0])
    update(subtree1.keys(),tree_dict)
    #print('root1', root1)

    print('subtree1:', tree_dict)

    # m = Mutation(Individual1,0.9)
    # m.do_mutation()
