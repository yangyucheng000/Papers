from __future__ import division
import numpy as np
import random
class Selection(object):

    def RouletteSelection(self, _a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]#indx从大到小排序
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ +=sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index

    def tournament_selection(self,_a, k, elementNum=2):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]  # indx从大到小排序
        sort_a = a[idx]
        # 选择出的个体序号 列表
        selected_index = []
        # 对列表排序, 排序规则按个人需求定制,修改mycmp即可

        for i in range(k):
            tempList = random.sample(list(idx), elementNum)
            if list(idx).index(tempList[0])<=list(idx).index(tempList[1]):
                selected_index.append(tempList[0])
            else:
                selected_index.append(tempList[1])
        ###返回选择的索引列表
        return selected_index

    def Truncation_selection(self,_a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]  # indx从大到小排序
        # 选择出的个体序号 列表
        selected_index = []
        # 对列表排序, 排序规则按个人需求定制,修改mycmp即可
        for i in range(k):
            selected_index.append(list(idx)[i])

            ###返回选择的索引列表
        return selected_index



if __name__ == '__main__':
    s = Selection()
    a = [1, 3, 2, 1, 9, 1, 5]
    selected_index = s.Truncation_selection(a, k=4)

    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)