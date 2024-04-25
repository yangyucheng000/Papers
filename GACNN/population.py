import torch
import os
import torch.nn as nn
import collections
from resnet import resnet18

import random
import json
from torch.utils import data
from torch.autograd import Variable

import eval_dataset
from treelib import Tree, Node
import json
import copy
from multiprocessing import Pool
import time
keys_layers=[[],[],[],[],[],[],[],[]]
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
                keys_layers[0].append(temp_key)
            elif temp_key[-1]=='2':
                keys_layers[1].append(temp_key)
            elif temp_key[-1] == '3':
                keys_layers[2].append(temp_key)
            elif temp_key[-1] == '4':
                keys_layers[3].append(temp_key)
            elif temp_key[-1] == '5':
                keys_layers[4].append(temp_key)
            elif temp_key[-1] == '6':
                keys_layers[5].append(temp_key)
            elif temp_key[-1] == '7':
                keys_layers[6].append(temp_key)
            else:
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

class Population(object):
    def __init__(self, params, gen_no,kind):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = 60
        self.params = params
        self.individuals = []
        self.kind=kind

    def initialize(self):
        if self.kind==20:
            tree_ict = [{
                "5-1": {

                    #'12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}},
                    #'13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}},
                    '15-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}}},

                    '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}},
                             '33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}
                             },

                    '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}},
                             '49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}},
                             #'86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}
                             },

                    '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}},

                    '35-2': {'23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}},
                             '35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'35-8': None}}}}},
                             '60-3': {'60-4': {'60-5': {'60-6': {'60-7': {'60-8': None}}}}}
                             },

                    #'47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}},

                    '5-2': {'5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}},
                            '79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}},
                            #'84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}},
                            '89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}}
                            },

                    '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}},

                    '6-2': {'6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},

                    '63-2': {#'39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}},
                             '63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}},
                             #'94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}
                             },

                    '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}},
                    '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},

                    '82-2': {'82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}},

                    #'9-2': {'9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}},

                    # '91-2': {
                    #     '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}},
                    #     '91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}},
                    '92-2': {'92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}},
                    '93-2': {'93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}},

                }
            }]
        if self.kind==30:
            tree_ict = [{
                "5-1": {

                    '12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}},
                    '13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}},
                    '15-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}}},

                    '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}},
                             '33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}
                             },

                    '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}},
                             '49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}},
                             '86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}
                             },

                    '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}},

                    '35-2': {'23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}},
                             '35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'35-8': None}}}}},
                             '60-3': {'60-4': {'60-5': {'60-6': {'60-7': {'60-8': None}}}}}
                             },

                    '47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}},

                    '5-2': {'5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}},
                            '79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}},
                            '84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}},
                            '89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}}
                            },

                    '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}},

                    '6-2': {'6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},

                    '63-2': {'39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}},
                             '63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}},
                             '94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}
                             },

                    '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}},
                    '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},

                    '82-2': {'82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}},

                    '9-2': {'9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}},

                    '91-2': {
                        '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}},
                        '91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}},
                    '92-2': {'92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}},
                    '93-2': {'93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}},

                }
            }]
        if self.kind==60:
            tree_ict=[
            {"27-1": {
                '11-2': {'11-3': {'11-4': {'11-5': {'11-6': {'11-7': {'11-8': None}}}}}},
                 '12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}},
                 '13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}},
                '15-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}},
                         '54-3': {'54-4': {'54-5': {'54-6': {'54-7': {'54-8': None}}}}},
                         },
                '19-2': {'19-3': {'19-4': {'19-5': {'19-6': {'19-7': {'19-8': None}}}}}},

                 '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}},
                          '33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}
                          },

                 '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}},
                          '49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}},
                          '86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}
                          },

                 '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}},


                 '35-2': {'23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}},
                          '35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'35-8': None}}}}},
                          '60-3': {'60-4': {'60-5': {'60-6': {'60-7': {'60-8': None}}}}},
                          '70-3': {'70-4': {'70-5': {'70-6': {'70-7': {'70-8': None}}}}}
                          },

                '42-2': {'42-3': {'42-4': {'42-5': {'42-6': {'42-7': {'42-8': None}}}}}},

                 '47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}},


                 '5-2': {'17-3': {'17-4': {'17-5': {'17-6': {'17-7': {'17-8': None}}}}},
                         '20-3': {'20-4': {'20-5': {'20-6': {'20-7': {'20-8': None}}}}},
                         '40-3': {'40-4': {'40-5': {'40-6': {'40-7': {'40-8': None}}}}},
                     '5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}},
                         '53-3': {'53-4': {'53-5': {'53-6': {'53-7': {'53-8': None}}}}},
                         '79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}},
                         '8-3': {'8-4': {'8-5': {'8-6': {'8-7': {'8-8': None}}}}},
                         '84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}},
                         '89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}},
                         '95-3': {'95-4': {'95-5': {'95-6': {'95-7': {'95-8': None}}}}}
                         },

                 '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}},
                '57-2': {'57-3': {'57-4': {'57-5': {'57-6': {'57-7': {'57-8': None}}}}}},

                 '6-2': {'6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},
                '62-2': {'62-3': {'62-4': {'62-5': {'62-6': {'62-7': {'62-8': None}}}}}},


                 '63-2': {'37-3': {'37-4': {'37-5': {'37-6': {'37-7': {'37-8': None}}}}},
                     '39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}},
                          '63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}},
                          '94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}
                          },
                '65-2': {'65-3': {'65-4': {'65-5': {'65-6': {'65-7': {'65-8': None}}}}}},
                '66-2': {'66-3': {'66-4': {'66-5': {'66-6': {'66-7': {'66-8': None}}}}}},
                '68-2': {'68-3': {'68-4': {'68-5': {'68-6': {'68-7': {'68-8': None}}}}}},
                 '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}},
                 '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},
                '72-2': {'72-3': {'72-4': {'72-5': {'72-6': {'72-7': {'72-8': None}}}}}},
                '75-2': {'75-3': {'75-4': {'75-5': {'75-6': {'75-7': {'75-8': None}}}}}},
                '76-2': {'76-3': {'76-4': {'76-5': {'76-6': {'76-7': {'76-8': None}}}}}},

                 '82-2': {'82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}},


                 '9-2': { '21-3': {'21-4': {'21-5': {'21-6': {'21-7': {'21-8': None}}}}},
                          '59-3': {'59-4': {'59-5': {'59-6': {'59-7': {'59-8': None}}}}},
                          '77-3': {'77-4': {'77-5': {'77-6': {'77-7': {'77-8': None}}}}},
                          '83-3': {'83-4': {'83-5': {'83-6': {'83-7': {'83-8': None}}}}},
                     '9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}},
                '90-2': {'90-3': {'90-4': {'90-5': {'90-6': {'90-7': {'90-8': None}}}}}},
                 '91-2': {
                     '48-3': {'48-4': {'48-5': {'48-6': {'48-7': {'48-8': None}}}}},
                     '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}},
                     '87-3': {'87-4': {'87-5': {'87-6': {'87-7': {'87-8': None}}}}},
                     '91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}},
                 '92-2': {'14-3': {'14-4': {'14-5': {'14-6': {'14-7': {'14-8': None}}}}},
                          '38-3': {'38-4': {'38-5': {'38-6': {'38-7': {'38-8': None}}}}},
                     '92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}},
                 '93-2':{'93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}},
                '98-2': {'98-3': {'98-4': {'98-5': {'98-6': {'98-7': {'98-8': None}}}}}},


            }}


        ]
        if self.kind==100:
            tree_ict = [
                {"95-1": {
                    '11-2': {'11-3': {'11-4': {'11-5': {'11-6': {'11-7': {'11-8': None}}}}}},
                    '12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}},
                    '13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}},
                    '15-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}},
                             '54-3': {'54-4': {'54-5': {'54-6': {'54-7': {'54-8': None}}}}},
                             },
                    '19-2': {'19-3': {'19-4': {'19-5': {'19-6': {'19-7': {'19-8': None}}}}}},

                    '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}},
                             '33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}
                             },

                    '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}},
                             '49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}},
                             '86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}
                             },
                    '30-2': {'30-3': {'30-4': {'30-5': {'30-6': {'30-7': {'30-8': None}}}}}},
                    '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}},
                    '34-2': {'34-3': {'34-4': {'34-5': {'34-6': {'34-7': {'34-8': None}}}}}},
                    '35-2': {'2-3': {'2-4': {'2-5': {'2-6': {'2-7': {'2-8': None}}}}},
                             '23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}},
                             '35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'35-8': None}}}}},
                             '44-3': {'44-4': {'44-5': {'44-6': {'44-7': {'44-8': None}}}}},
                             '60-3': {'60-4': {'60-5': {'60-6': {'60-7': {'60-8': None}}}}},
                             '70-3': {'70-4': {'70-5': {'70-6': {'70-7': {'70-8': None}}}}}
                             },
                    '36-2': {'36-3': {'36-4': {'36-5': {'36-6': {'36-7': {'36-8': None}}}}}},
                    '42-2': {'42-3': {'42-4': {'42-5': {'42-6': {'42-7': {'42-8': None}}}}}},
                    '45-2': {'45-3': {'45-4': {'45-5': {'45-6': {'45-7': {'45-8': None}}}}}},

                    '47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}},

                    '5-2': {'16-3': {'16-4': {'16-5': {'16-6': {'16-7': {'16-8': None}}}}},
                        '17-3': {'17-4': {'17-5': {'17-6': {'17-7': {'17-8': None}}}}},
                            '20-3': {'20-4': {'20-5': {'20-6': {'20-7': {'20-8': None}}}}},
                            '28-3': {'28-4': {'28-5': {'28-6': {'28-7': {'28-8': None}}}}},
                            '29-3': {'29-4': {'29-5': {'29-6': {'29-7': {'29-8': None}}}}},
                            '32-3': {'32-4': {'32-5': {'32-6': {'32-7': {'32-8': None}}}}},
                            '40-3': {'40-4': {'40-5': {'40-6': {'40-7': {'40-8': None}}}}},
                            '41-3': {'41-4': {'41-5': {'41-6': {'41-7': {'41-8': None}}}}},
                            '43-3': {'43-4': {'43-5': {'43-6': {'43-7': {'43-8': None}}}}},
                            '5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}},
                            '51-3': {'51-4': {'51-5': {'51-6': {'51-7': {'51-8': None}}}}},
                            '53-3': {'53-4': {'53-5': {'53-6': {'53-7': {'53-8': None}}}}},
                            '61-3': {'61-4': {'61-5': {'61-6': {'61-7': {'61-8': None}}}}},
                            '67-3': {'67-4': {'67-5': {'67-6': {'67-7': {'67-8': None}}}}},
                            '71-3': {'71-4': {'71-5': {'71-6': {'71-7': {'71-8': None}}}}},
                            '79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}},
                            '8-3': {'8-4': {'8-5': {'8-6': {'8-7': {'8-8': None}}}}},
                            '84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}},
                            '85-3': {'85-4': {'85-5': {'85-6': {'85-7': {'85-8': None}}}}},
                            '89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}},
                            '95-3': {'95-4': {'95-5': {'95-6': {'95-7': {'95-8': None}}}}},
                            '97-3': {'97-4': {'97-5': {'97-6': {'97-7': {'97-8': None}}}}},
                            '99-3': {'99-4': {'99-5': {'99-6': {'99-7': {'99-8': None}}}}}
                            },
                    '50-2': { '0-3': {'0-4': {'0-5': {'0-6': {'0-7': {'0-8': None}}}}},
                        '50-3': {'50-4': {'50-5': {'50-6': {'50-7': {'50-8': None}}}}}},
                    '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}},
                    '57-2': {'18-3': {'18-4': {'18-5': {'18-6': {'18-7': {'18-8': None}}}}},
                        '57-3': {'57-4': {'57-5': {'57-6': {'57-7': {'57-8': None}}}}}},

                    '6-2': {'55-3': {'55-4': {'55-5': {'55-6': {'55-7': {'55-8': None}}}}},
                        '6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},
                    '62-2': {'25-3': {'25-4': {'25-5': {'25-6': {'25-7': {'25-8': None}}}}},
                             '52-3': {'52-4': {'52-5': {'52-6': {'52-7': {'52-8': None}}}}},
                        '62-3': {'62-4': {'62-5': {'62-6': {'62-7': {'62-8': None}}}}}},

                    '63-2': {'10-3': {'10-4': {'10-5': {'10-6': {'10-7': {'10-8': None}}}}},
                        '37-3': {'37-4': {'37-5': {'37-6': {'37-7': {'37-8': None}}}}},
                             '39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}},
                             '63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}},
                             '73-3': {'73-4': {'73-5': {'73-6': {'73-7': {'73-8': None}}}}},
                             '94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}
                             },
                    '64-2': {'64-3': {'64-4': {'64-5': {'64-6': {'64-7': {'64-8': None}}}}}},
                    '65-2': {'65-3': {'65-4': {'65-5': {'65-6': {'65-7': {'65-8': None}}}}},
                             '96-3': {'96-4': {'96-5': {'96-6': {'96-7': {'96-8': None}}}}},
                             },
                    '66-2': {'66-3': {'66-4': {'66-5': {'66-6': {'66-7': {'66-8': None}}}}}},
                    '68-2': {'68-3': {'68-4': {'68-5': {'68-6': {'68-7': {'68-8': None}}}}}},
                    '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}},
                    '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},
                    '72-2': {'72-3': {'72-4': {'72-5': {'72-6': {'72-7': {'72-8': None}}}}}},
                    '75-2': {'75-3': {'75-4': {'75-5': {'75-6': {'75-7': {'75-8': None}}}}}},
                    '76-2': {'76-3': {'76-4': {'76-5': {'76-6': {'76-7': {'76-8': None}}}}}},
                    '80-2': {'80-3': {'80-4': {'80-5': {'80-6': {'80-7': {'80-8': None}}}}}},
                    '82-2': { '26-3': {'26-4': {'26-5': {'26-6': {'26-7': {'26-8': None}}}}},
                        '82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}},

                    '9-2': {'1-3': {'1-4': {'1-5': {'1-6': {'1-7': {'1-8': None}}}}},
                        '21-3': {'21-4': {'21-5': {'21-6': {'21-7': {'21-8': None}}}}},
                            '22-3': {'22-4': {'22-5': {'22-6': {'22-7': {'22-8': None}}}}},
                            '4-3': {'4-4': {'4-5': {'4-6': {'4-7': {'4-8': None}}}}},
                            '46-3': {'46-4': {'46-5': {'46-6': {'46-7': {'46-8': None}}}}},
                            '59-3': {'59-4': {'59-5': {'59-6': {'59-7': {'59-8': None}}}}},
                            '77-3': {'77-4': {'77-5': {'77-6': {'77-7': {'77-8': None}}}}},
                            '81-3': {'81-4': {'81-5': {'81-6': {'81-7': {'81-8': None}}}}},
                            '83-3': {'83-4': {'83-5': {'83-6': {'83-7': {'83-8': None}}}}},
                            '9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}},
                    '90-2': {'90-3': {'90-4': {'90-5': {'90-6': {'90-7': {'90-8': None}}}}}},
                    '91-2': {'3-3': {'3-4': {'3-5': {'3-6': {'3-7': {'3-8': None}}}}},
                        '48-3': {'48-4': {'48-5': {'48-6': {'48-7': {'48-8': None}}}}},
                             '58-3': {'58-4': {'58-5': {'58-6': {'58-7': {'58-8': None}}}}},
                        '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}},
                             '78-3': {'78-4': {'78-5': {'78-6': {'78-7': {'78-8': None}}}}},
                        '87-3': {'87-4': {'87-5': {'87-6': {'87-7': {'87-8': None}}}}},
                             '88-3': {'88-4': {'88-5': {'88-6': {'88-7': {'88-8': None}}}}},
                        '91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}},
                    '92-2': {'14-3': {'14-4': {'14-5': {'14-6': {'14-7': {'14-8': None}}}}},
                             '38-3': {'38-4': {'38-5': {'38-6': {'38-7': {'38-8': None}}}}},
                             '92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}},
                    '93-2': {'93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}},
                    '98-2': {'98-3': {'98-4': {'98-5': {'98-6': {'98-7': {'98-8': None}}}}}},

                }}

            ]



        for i in range(0,1):
            indi_no = 'indi%02d%02d' % (0, i)
            indi=Individual(self.params, indi_no,self.kind,nee1=True)
            indi.tree_dict=tree_ict[i]
            indi.paths=indi.dg(indi.tree_dict)
            self.individuals.append(indi)
            self.number_id += 1
        for _ in range(1,self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no,self.kind)
            #print(indi.tree_dict)
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.indi_no = indi_no
            self.number_id += 1
            #indi.number_id = 8
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)

class Individual(object):
    def __init__(self, n,indi_no,kind,nee1=False):
        self.kind=kind
        #print(self.kind)
        self.paths=[]
        self.indi_no=indi_no
        self.tree_dict={}
        self.acc = -1.0
        self.l12_list = []  # 反回的数据
        self.qwq = []
        self.fitvalue=0
        self.key_num=0
        if nee1==False:
            self.tree_dict=self.create_tree(self.kind)
        self.paths = self.dg(self.tree_dict)

    def reset_acc(self):
        self.acc = -1.0

    def dg(self, data):
        # global qwq
        l1_list = []
        if data==None:
            return None
        for i in data.keys():
            if isinstance(data[i], dict):
                self.qwq.append(i)
                self.dg(data[i])
                self.qwq = self.qwq[:-1]
            else:
                l1_list.append(i)
        if l1_list:
            for qwq1_i in l1_list:
                self.l12_list.append(self.qwq + [qwq1_i])
        return self.l12_list




    def print_dict_(self, text, prefix='\t'):
        for i in text.keys():
            if isinstance(text[i], dict):
                print(prefix + '+ ' + i)
                self.print_dict_(text[i], prefix + '|   ')  # 最后一个└
            else:
                print(prefix + '- ' + i)

    def create_tree(self, n):
        path_num = 0
        tree_dict = {}
        selected=[]
        #order = random.sample(range(0, 100), 100)
        #tree_dict = json.loads(json.dumps(tree_dict))
        order=[5, 24, 92, 35, 63, 56, 82, 15, 6, 27, 23, 79, 31, 69, 49,
               60, 89, 7, 93, 33, 12, 91, 9, 94, 47, 86, 74, 39, 13, 84, 72,
               70, 77, 83, 87, 17, 48, 68, 75, 90, 59, 19, 42, 21, 14, 76, 66,
               57, 95, 54, 11, 98, 53, 38, 65, 37, 8, 40, 20, 62, 4, 99, 16, 96,
               61, 32, 78, 50, 29, 18, 2, 0, 58, 44, 30, 88, 3, 52, 73, 36, 43, 28,
               1, 67, 10, 41, 22, 97, 55, 25, 46, 64, 34, 45, 81, 85, 71, 80, 51, 26]
        if self.kind == 20:
            order = order[:20]
        elif self.kind == 30:
            order = order[:30]
        elif self.kind == 40:
            order= order[:40]
        elif self.kind == 60:
            order = order[:60]
        elif self.kind == 80:
            order = order[:80]
        root = str(random.choice(order)) + "-1"

        num2 = random.randint(3, 6)  # 根的子节点个数，也是第二层所有节点个数
        layer2 = {}
        for i1 in range(num2):
            block2 = {}
            #block2 = json.loads(json.dumps(block2))
            key2 = str(random.choice(order))  + '-2'
            block2.setdefault(key2)
            layer2.update(block2)
            tree_dict[root] = layer2  # 链接到根节点

            num3 = random.randint(3, 5)  # 第二层某个节点的子节点个数
            layer3 = {}  # 记录第二层某个节点的子节点
            for i2 in range(num3):
                block3 = {}
                #block3 = json.loads(json.dumps(block3))
                key3 =str(random.choice(order))  + '-3'
                block3.setdefault(key3)
                layer3.update(block3)
                # layer2_all.update(block)
                tree_dict[root][key2] = layer3

                # layer3_all = {}#记录第四层所有节点

                # print(i2)
                num4 = random.randint(3, 5)  # 第三层的某个节点的子节点个数
                layer4 = {}
                for i3 in range(num4):
                    block4 = {}
                    #block4 = json.loads(json.dumps(block4))
                    key4 =str(random.choice(order))  + '-4'
                    # block[key] = ''
                    block4.setdefault(key4)
                    layer4.update(block4)
                    # layer3_all.update(block)
                    tree_dict[root][key2][key3] = layer4

                    num5 = random.randint(3, 5)  # 第4层的某个节点的子节点个数
                    layer5 = {}
                    for i4 in range(num5):
                        block5 = {}
                        #block5 = json.loads(json.dumps(block5))
                        key5 =str(random.choice(order))  + '-5'
                        # block[key] = ''
                        block5.setdefault(key5)
                        layer5.update(block5)
                        # layer3_all.update(block)
                        tree_dict[root][key2][key3][key4] = layer5

                        # print(i2)
                        num6 = random.randint(1, 5)  # 第5层的某个节点的子节点个数
                        layer6 = {}
                        for i5 in range(num6):
                            block6 = {}
                            #block6 = json.loads(json.dumps(block6))
                            key6 = str(random.choice(order))  + '-6'
                            # block[key] = ''
                            block6.setdefault(key6)
                            layer6.update(block6)
                            # layer3_all.update(block)
                            tree_dict[root][key2][key3][key4][key5] = layer6

                            # print(i2)
                            num7 = random.randint(1, 5)  # 第6层的某个节点的子节点个数
                            layer7 = {}
                            for i6 in range(num7):
                                block7 = {}
                                #block7 = json.loads(json.dumps(block7))
                                key7 = str(random.choice(order))  + '-7'
                                # block[key] = ''
                                block7.setdefault(key7)
                                layer7.update(block7)
                                # layer3_all.update(block)
                                tree_dict[root][key2][key3][key4][key5][key6] = layer7

                                #num8 = random.randint(1, 2)  # 第7层的某个节点的子节点个数
                                num8=1
                                # print('num8:',num8)
                                layer8 = {}
                                for i7 in range(num8):
                                    block8 = {}
                                    #block8 = json.loads(json.dumps(block8))
                                    key8 = str(random.choice(order))  + '-8'
                                    if key8 in selected:
                                        key8 = str(random.choice(order)) + '-8'
                                        while key8 in selected and len(selected)<self.kind:
                                            key8 = str(random.choice(order)) + '-8'
                                            #print(key8)
                                    selected.append(key8)

                                    block8.setdefault(key8)
                                    layer8.update(block8)
                                    # layer3_all.update(block)
                                    tree_dict[root][key2][key3][key4][key5][key6][key7] = layer8

                                    path_num = path_num + 1
                                    # print(path_num)
                                    # print(layer8)
                                    if path_num > self.kind+5:
                                        return tree_dict

    def eval_paths(self, indi_no):
        model_path = './model/'
        nums = len(self.paths)
        newpath = './populations_%2d/models_'%(self.kind) + indi_no + '/'
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
        # print(nums)
        root=['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
        'bn1.num_batches_tracked']
        block1 = [
                  'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                  'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight',
                  'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var',
                  'layer1.0.bn2.num_batches_tracked']
        block2 = ['layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean',
                  'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight',
                  'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                  'layer1.1.bn2.num_batches_tracked']
        block3 = ['layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean',
                  'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                  'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
                  'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight',
                  'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean',
                  'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked']
        block4 = ['layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
                  'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
                  'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
                  'layer2.1.bn2.num_batches_tracked']
        block5 = ['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean',
                  'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight',
                  'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
                  'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight',
                  'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean',
                  'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked']
        block6 = ['layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
                  'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight',
                  'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
                  'layer3.1.bn2.num_batches_tracked']
        block7 = ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean',
                  'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight',
                  'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var',
                  'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight',
                  'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean',
                  'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked']
        block8 = ['layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean',
                  'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight',
                  'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var',
                  'layer4.1.bn2.num_batches_tracked', 'fc.weight','fc.bias']

        # print(nums)
        order=[]
        #print(self.kind)
        for i in range(nums):#遍历所有路径
            #print(i)
            if i >=self.kind:
                break
            path = self.paths[i]
            num = len(path)
            dict = torch.load('./model/model_1.pth')
            used_block = []
            lastblock=path[-1]

            #print(kindblock)
            order.append(int(lastblock[:-2]))
            #print(lastblock)

            root_path = model_path + 'model_' +lastblock[:-2] + '.pth'
            trained_weight1 = torch.load(root_path)
            for j in range(len(root)):
                dict[root[j]] = trained_weight1[root[j]]
            for k in range(num):#遍历一条路径上的所有块
                block = path[k]
                weigth_path = model_path + 'model_' + block[:-2] + '.pth'
                # print(weigth_path)
                trained_weight = torch.load(weigth_path)
                #trained_weight1 = torch.load(root_path)
                if block[-1] == '1':
                    used_block = block1
                elif block[-1] == '2':
                    used_block = block2
                elif block[-1] == '3':
                    used_block = block3
                elif block[-1] == '4':
                    used_block = block4
                elif block[-1] == '5':
                    used_block = block5
                elif block[-1] == '6':
                    used_block = block6
                elif block[-1] == '7':
                    used_block = block7
                elif block[-1] == '8':
                    used_block = block8

                for j in range(len(used_block)):

                    dict[used_block[j]] = trained_weight[used_block[j]]

            torch.save(dict, newpath + 'model_' + lastblock[:-2] + '.pth')

        is_support = torch.cuda.is_available()
        if is_support:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # print(len(order))
        # print(order)
        order1 = [5, 24, 92, 35, 63, 56, 82, 15, 6, 27, 23, 79, 31, 69, 49, 60, 89, 7, 93, 33, 12, 91, 9, 94, 47, 86, 74,
                 39, 13, 84, 72, 70, 77, 83, 87, 17, 48, 68, 75, 90, 59, 19, 42, 21, 14, 76, 66, 57, 95, 54, 11, 98, 53,
                 38, 65, 37, 8, 40, 20, 62, 4, 99, 16, 96, 61, 32, 78, 50, 29, 18, 2, 0, 58, 44, 30, 88, 3, 52, 73, 36,
                 43, 28, 1, 67, 10, 41, 22, 97, 55, 25, 46, 64, 34, 45, 81, 85, 71, 80, 51, 26]
        if self.kind==20:
            order1 = order1[:20]
        elif self.kind == 30:
            order1 = order1[:30]
        elif self.kind==40:
            order1=order1[:40]
        elif self.kind==60:
            order1=order1[:60]
        elif self.kind==80:
            order1=order1[:80]
        test_data = eval_dataset.MyDataset1(resize=224,order=order,task_num=len(order))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
        order=list(set(order))
        print(len(order))
        # print(len(order1))
        n=len(order1)-len(order)
        print(n)
        model = NewModel(newpath,order)

        model.to(device)

        correct_sample = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                batch_x = data['data']
                batch_y = data['label']
                batch_x = Variable(batch_x).float().to(device)
                batch_y = Variable(batch_y).long().to(device)
                # print("inputs", batch_x.data.size(), "labels", batch_y.data)
                out = model(batch_x)

                correct_sample += (torch.Tensor(out).cpu() == batch_y.cpu()).sum().item()
                total += batch_y.size(0)
        keys1 = get_dict_allkeys(self.tree_dict)
        self.key_num=len(keys1)

        print(len(keys1))

        self.acc = correct_sample / (total+n*100)

        self.fitvalue =self.acc
        keys_layers[0].clear()
        keys_layers[1].clear()
        keys_layers[2].clear()
        keys_layers[3].clear()
        keys_layers[4].clear()
        keys_layers[5].clear()
        keys_layers[6].clear()
        keys_layers[7].clear()
        key_list.clear()

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.indi_no))
        _str.append('Acc:%.5f' % (self.acc))
        _str.append('paths:%s' % (self.paths))
        _str.append('fit_value:%s' % (self.fitvalue))
        return '\n'.join(_str)




class NewModel(nn.Module):
    def __init__(self,newpath,order=None):
        super(NewModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allmodel = {}
        self.trees={}
        self.totalkeys = {}
       # print(1)
        self.order=order
        # self.order = random.sample(range(0, 10), 10)  # 随机生成顺序
        #print(self.order)
        self.initmodels(newpath)
        # collections.OrderedDict()

    def initmodels(self,newpath):  # 修改合并网络及网络存放地址
        path = newpath
        dir = os.listdir(path)

        k = 0
        for o in self.order:
            for i in dir:
                j = i[:-4]
                #print(j)
                if j[6:] == str(o):
                    model = resnet18(num_classes=2)
                    model.cuda()
                    model.eval()
                    trained_weight = torch.load(path + i)
                    model.load_state_dict(trained_weight)
                    self.allmodel[k] = model

                    self.initmodel(modelpath=path + 'model_' + str(o) + '.pth', modelkind=k)
                    k=k+1
                    break
        #print(self.trees[k-1])


    def initmodel(self, modelpath, modelkind):
        model = resnet18(num_classes=2)
        trained_weight = torch.load(modelpath)
        model.load_state_dict(trained_weight)
        #model = myvgg.get_trained_vgg(path=modelpath)
        self.cuda()
        model.cuda()
        model.eval()
        self.eval()



    def forward(self, x):
        res = {}
        for k in self.allmodel.keys():
            res[k] = self.allmodel[k](x)  # all path results
            res[k]= res[k].softmax(1)
        ans = self.parseres(res)
        return ans

    def parseres(self, res):
        key = []
        for i in res.keys():
            key.append(i)

        result = [-1 for i in range(res[key[0]].shape[0])]

        for i in range(res[key[0]].shape[0]):
            max = -100
            max1=-100
            result[i] = 0
            for k in res.keys():
                if res[k][i][1] > 0 and res[k][i][1]  > max: # 对于正数的结果选差值最大的,结果0.6455
                    max=res[k][i][1]
                    result[i]=self.order[k]

        return result


if __name__ == '__main__':
  #tree_dict={'4-1': {'1-2': {'1-3': {'1-4': {'1-5': {'1-6': {'1-7': {'1-8': None}}}}}}, '10-2': {'10-3': {'10-4': {'10-5': {'10-6': {'10-7': {'10-8': None}}}}}}, '11-2': {'11-3': {'11-4': {'11-5': {'11-6': {'11-7': {'11-8': None}}}}}}, '12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}}, '13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}}, '14-2': {'14-3': {'14-4': {'14-5': {'14-6': {'14-7': {'14-8': None}}}}}}, '16-2': {'16-3': {'16-4': {'16-5': {'16-6': {'16-7': {'16-8': None}}}}}}, '17-2': {'17-3': {'17-4': {'17-5': {'17-6': {'17-7': {'17-8': None}}}}}}, '18-2': {'18-3': {'18-4': {'18-5': {'18-6': {'18-7': {'18-8': None}}}}}}, '19-2': {'19-3': {'19-4': {'19-5': {'19-6': {'19-7': {'19-8': None}}}}}, '3-3': {'3-4': {'3-5': {'3-6': {'3-7': {'3-8': None}}}}}, '53-3': {'53-4': {'53-5': {'53-6': {'53-7': {'53-8': None}}}}}}, '21-2': {'21-3': {'21-4': {'21-5': {'21-6': {'21-7': {'21-8': None}}}}}}, '23-2': {'23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}}}, '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}}}, '26-2': {'26-3': {'26-4': {'26-5': {'26-6': {'26-7': {'26-8': None}}}}}}, '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}}}, '28-2': {'28-3': {'28-4': {'28-5': {'28-6': {'28-7': {'28-8': None}}}}}}, '30-2': {'30-3': {'30-4': {'30-5': {'30-6': {'30-7': {'30-8': None}}}}}}, '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}}, '33-2': {'33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}}, '34-2': {'34-3': {'34-4': {'34-5': {'34-6': {'34-7': {'34-8': None}}}}}}, '35-2': {'35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'99-8': None}}}}}}, '36-2': {'36-3': {'36-4': {'36-5': {'36-6': {'36-7': {'36-8': None}}}}}}, '37-2': {'37-3': {'37-4': {'37-5': {'37-6': {'37-7': {'37-8': None}}}}}}, '38-2': {'0-3': {'0-4': {'0-5': {'0-6': {'0-7': {'0-8': None}}}}}, '38-3': {'38-4': {'38-5': {'38-6': {'38-7': {'38-8': None}}}}}}, '39-2': {'39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}}}, '40-2': {'40-3': {'40-4': {'40-5': {'40-6': {'40-7': {'40-8': None}}}}}}, '42-2': {'42-3': {'42-4': {'42-5': {'42-6': {'42-7': {'42-8': None}}}}}}, '43-2': {'43-3': {'43-4': {'43-5': {'43-6': {'43-7': {'43-8': None}}}}}}, '44-2': {'44-3': {'44-4': {'44-5': {'44-6': {'44-7': {'44-8': None}}}}}}, '45-2': {'45-3': {'45-4': {'45-5': {'45-6': {'45-7': {'45-8': None}}}}}}, '46-2': {'46-3': {'46-4': {'46-5': {'46-6': {'46-7': {'46-8': None}}}}}}, '47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}}, '48-2': {'48-3': {'48-4': {'48-5': {'48-6': {'48-7': {'48-8': None}}}}}, '97-3': {'97-4': {'97-5': {'97-6': {'97-7': {'97-8': None}}}}}}, '49-2': {'49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}}}, '5-2': {'5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}}}, '50-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}}, '32-3': {'32-4': {'32-5': {'32-6': {'32-7': {'32-8': None}}}}}, '41-3': {'41-4': {'41-5': {'41-6': {'41-7': {'41-8': None}}}}}, '50-3': {'50-4': {'50-5': {'50-6': {'50-7': {'50-8': None}}}}}, '66-3': {'66-4': {'66-5': {'66-6': {'66-7': {'66-8': None}}}}}, '87-3': {'87-4': {'87-5': {'87-6': {'87-7': {'87-8': None}}}}}, '93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}, '95-3': {'95-4': {'95-5': {'95-6': {'95-7': {'95-8': None}}}}}}, '51-2': {'51-3': {'51-4': {'51-5': {'51-6': {'51-7': {'51-8': None}}}}}}, '52-2': {'52-3': {'52-4': {'52-5': {'52-6': {'52-7': {'52-8': None}}}}}}, '54-2': {'54-3': {'54-4': {'54-5': {'54-6': {'54-7': {'54-8': None}}}}}}, '55-2': {'55-3': {'55-4': {'55-5': {'55-6': {'55-7': {'55-8': None}}}}}}, '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}}, '57-2': {'57-3': {'57-4': {'57-5': {'57-6': {'57-7': {'57-8': None}}}}}}, '58-2': {'58-3': {'58-4': {'58-5': {'58-6': {'58-7': {'58-8': None}}}}}}, '59-2': {'59-3': {'59-4': {'59-5': {'59-6': {'59-7': {'59-8': None}}}}}}, '6-2': {'6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}}, '60-2': {'60-3': {'60-4': {'60-5': {'60-6': {'70-7': {'70-8': None}}}}}}, '61-2': {'61-3': {'61-4': {'61-5': {'61-6': {'61-7': {'61-8': None}}}}}}, '62-2': {'62-3': {'62-4': {'62-5': {'62-6': {'62-7': {'62-8': None}}}}}}, '63-2': {'63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}}}, '64-2': {'64-3': {'64-4': {'64-5': {'64-6': {'64-7': {'64-8': None}}}}}}, '65-2': {'25-3': {'25-4': {'25-5': {'25-6': {'25-7': {'25-8': None}}}}}, '65-3': {'65-4': {'65-5': {'65-6': {'65-7': {'65-8': None}}}}}}, '67-2': {'67-3': {'67-4': {'67-5': {'67-6': {'67-7': {'67-8': None}}}}}}, '68-2': {'68-3': {'68-4': {'68-5': {'68-6': {'68-7': {'68-8': None}}}}}}, '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}}, '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}}, '70-2': {'70-3': {'70-4': {'70-5': {'70-6': {'70-7': {'70-8': None}}}}}}, '71-2': {'71-3': {'71-4': {'71-5': {'71-6': {'71-7': {'71-8': None}}}}}}, '72-2': {'72-3': {'72-4': {'72-5': {'72-6': {'72-7': {'72-8': None}}}}}}, '73-2': {'73-3': {'73-4': {'73-5': {'73-6': {'73-7': {'73-8': None}}}}}}, '74-2': {'20-3': {'20-4': {'20-5': {'20-6': {'20-7': {'20-8': None}}}}}, '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}}}, '75-2': {'75-3': {'75-4': {'75-5': {'75-6': {'75-7': {'75-8': None}}}}}}, '76-2': {'76-3': {'76-4': {'76-5': {'76-6': {'76-7': {'76-8': None}}}}}}, '77-2': {'77-3': {'77-4': {'77-5': {'77-6': {'77-7': {'77-8': None}}}}}}, '78-2': {'78-3': {'78-4': {'78-5': {'78-6': {'78-7': {'78-8': None}}}}}}, '79-2': {'79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}}}, '8-2': {'8-3': {'8-4': {'8-5': {'8-6': {'8-7': {'8-8': None}}}}}}, '80-2': {'80-3': {'80-4': {'80-5': {'80-6': {'80-7': {'80-8': None}}}}}}, '82-2': {'82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}}, '83-2': {'83-3': {'83-4': {'83-5': {'83-6': {'83-7': {'83-8': None}}}}}}, '84-2': {'22-3': {'22-4': {'22-5': {'22-6': {'22-7': {'22-8': None}}}}}, '29-3': {'29-4': {'29-5': {'29-6': {'29-7': {'29-8': None}}}}}, '84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}}}, '85-2': {'85-3': {'85-4': {'85-5': {'85-6': {'85-7': {'85-8': None}}}}}}, '86-2': {'86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}}, '88-2': {'88-3': {'88-4': {'88-5': {'88-6': {'88-7': {'88-8': None}}}}}}, '89-2': {'89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}}}, '9-2': {'9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}}, '90-2': {'90-3': {'90-4': {'90-5': {'90-6': {'90-7': {'90-8': None}}}}}}, '91-2': {'91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}}, '92-2': {'92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}}, '93-2': {'57-3': {'57-4': {'57-5': {'57-6': {'57-7': {'57-8': None}}}}}}, '94-2': {'2-3': {'2-4': {'2-5': {'2-6': {'2-7': {'2-8': None}}}}}, '4-3': {'4-4': {'4-5': {'4-6': {'4-7': {'4-8': None}}}}}, '81-3': {'81-4': {'81-5': {'81-6': {'81-7': {'81-8': None}}}}}, '94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}}, '96-2': {'96-3': {'96-4': {'96-5': {'96-6': {'96-7': {'96-8': None}}}}}}, '98-2': {'98-3': {'98-4': {'98-5': {'98-6': {'98-7': {'98-8': None}}}}}}, '99-2': {'99-3': {'99-4': {'99-5': {'94-6': {'94-7': {'94-8': None}}}}}}}}

    tree_dict={"48-1": {
                    '11-2': {'11-3': {'11-4': {'11-5': {'11-6': {'11-7': {'11-8': None}}}}}},
                    '12-2': {'12-3': {'12-4': {'12-5': {'12-6': {'12-7': {'12-8': None}}}}}},
                    '13-2': {'13-3': {'13-4': {'13-5': {'13-6': {'13-7': {'13-8': None}}}}}},
                    '15-2': {'15-3': {'15-4': {'15-5': {'15-6': {'15-7': {'15-8': None}}}}},
                             '54-3': {'54-4': {'54-5': {'54-6': {'54-7': {'54-8': None}}}}},
                             },
                    '19-2': {'19-3': {'19-4': {'19-5': {'19-6': {'19-7': {'19-8': None}}}}}},

                    '24-2': {'24-3': {'24-4': {'24-5': {'24-6': {'24-7': {'24-8': None}}}}},
                             '33-3': {'33-4': {'33-5': {'33-6': {'33-7': {'33-8': None}}}}}
                             },

                    '27-2': {'27-3': {'27-4': {'27-5': {'27-6': {'27-7': {'27-8': None}}}}},
                             '49-3': {'49-4': {'49-5': {'49-6': {'49-7': {'49-8': None}}}}},
                             '86-3': {'86-4': {'86-5': {'86-6': {'86-7': {'86-8': None}}}}}
                             },
                    '30-2': {'30-3': {'30-4': {'30-5': {'30-6': {'30-7': {'30-8': None}}}}}},
                    '31-2': {'31-3': {'31-4': {'31-5': {'31-6': {'31-7': {'31-8': None}}}}}},
                    '34-2': {'34-3': {'34-4': {'34-5': {'34-6': {'34-7': {'34-8': None}}}}}},
                    '35-2': {'2-3': {'2-4': {'2-5': {'2-6': {'2-7': {'2-8': None}}}}},
                             '23-3': {'23-4': {'23-5': {'23-6': {'23-7': {'23-8': None}}}}},
                             '35-3': {'35-4': {'35-5': {'35-6': {'35-7': {'35-8': None}}}}},
                             '44-3': {'44-4': {'44-5': {'44-6': {'44-7': {'44-8': None}}}}},
                             '60-3': {'60-4': {'60-5': {'60-6': {'60-7': {'60-8': None}}}}},
                             '70-3': {'70-4': {'70-5': {'70-6': {'70-7': {'70-8': None}}}}}
                             },
                    '36-2': {'36-3': {'36-4': {'36-5': {'36-6': {'36-7': {'36-8': None}}}}}},
                    '42-2': {'42-3': {'42-4': {'42-5': {'42-6': {'42-7': {'42-8': None}}}}}},
                    '45-2': {'45-3': {'45-4': {'45-5': {'45-6': {'45-7': {'45-8': None}}}}}},

                    '47-2': {'47-3': {'47-4': {'47-5': {'47-6': {'47-7': {'47-8': None}}}}}},

                    '5-2': {'16-3': {'16-4': {'16-5': {'16-6': {'16-7': {'16-8': None}}}}},
                        '17-3': {'17-4': {'17-5': {'17-6': {'17-7': {'17-8': None}}}}},
                            '20-3': {'20-4': {'20-5': {'20-6': {'20-7': {'20-8': None}}}}},
                            '28-3': {'28-4': {'28-5': {'28-6': {'28-7': {'28-8': None}}}}},
                            '29-3': {'29-4': {'29-5': {'29-6': {'29-7': {'29-8': None}}}}},
                            '32-3': {'32-4': {'32-5': {'32-6': {'32-7': {'32-8': None}}}}},
                            '40-3': {'40-4': {'40-5': {'40-6': {'40-7': {'40-8': None}}}}},
                            '41-3': {'41-4': {'41-5': {'41-6': {'41-7': {'41-8': None}}}}},
                            '43-3': {'43-4': {'43-5': {'43-6': {'43-7': {'43-8': None}}}}},
                            '5-3': {'5-4': {'5-5': {'5-6': {'5-7': {'5-8': None}}}}},
                            '51-3': {'51-4': {'51-5': {'51-6': {'51-7': {'51-8': None}}}}},
                            '53-3': {'53-4': {'53-5': {'53-6': {'53-7': {'53-8': None}}}}},
                            '61-3': {'61-4': {'61-5': {'61-6': {'61-7': {'61-8': None}}}}},
                            '67-3': {'67-4': {'67-5': {'67-6': {'67-7': {'67-8': None}}}}},
                            '71-3': {'71-4': {'71-5': {'71-6': {'71-7': {'71-8': None}}}}},
                            '79-3': {'79-4': {'79-5': {'79-6': {'79-7': {'79-8': None}}}}},
                            '8-3': {'8-4': {'8-5': {'8-6': {'8-7': {'8-8': None}}}}},
                            '84-3': {'84-4': {'84-5': {'84-6': {'84-7': {'84-8': None}}}}},
                            '85-3': {'85-4': {'85-5': {'85-6': {'85-7': {'85-8': None}}}}},
                            '89-3': {'89-4': {'89-5': {'89-6': {'89-7': {'89-8': None}}}}},
                            '95-3': {'95-4': {'95-5': {'95-6': {'95-7': {'95-8': None}}}}},
                            '97-3': {'97-4': {'97-5': {'97-6': {'97-7': {'97-8': None}}}}},
                            '99-3': {'99-4': {'99-5': {'99-6': {'99-7': {'99-8': None}}}}}
                            },
                    '50-2': { '0-3': {'0-4': {'0-5': {'0-6': {'0-7': {'0-8': None}}}}},
                        '50-3': {'50-4': {'50-5': {'50-6': {'50-7': {'50-8': None}}}}}},
                    '56-2': {'56-3': {'56-4': {'56-5': {'56-6': {'56-7': {'56-8': None}}}}}},
                    '57-2': {'18-3': {'18-4': {'18-5': {'18-6': {'18-7': {'18-8': None}}}}},
                        '57-3': {'57-4': {'57-5': {'57-6': {'57-7': {'57-8': None}}}}}},

                    '6-2': {'55-3': {'55-4': {'55-5': {'55-6': {'55-7': {'55-8': None}}}}},
                        '6-3': {'6-4': {'6-5': {'6-6': {'6-7': {'6-8': None}}}}}},
                    '62-2': {'25-3': {'25-4': {'25-5': {'25-6': {'25-7': {'25-8': None}}}}},
                             '52-3': {'52-4': {'52-5': {'52-6': {'52-7': {'52-8': None}}}}},
                        '62-3': {'62-4': {'62-5': {'62-6': {'62-7': {'62-8': None}}}}}},

                    '63-2': {'10-3': {'10-4': {'10-5': {'10-6': {'10-7': {'10-8': None}}}}},
                        '37-3': {'37-4': {'37-5': {'37-6': {'37-7': {'37-8': None}}}}},
                             '39-3': {'39-4': {'39-5': {'39-6': {'39-7': {'39-8': None}}}}},
                             '63-3': {'63-4': {'63-5': {'63-6': {'63-7': {'63-8': None}}}}},
                             '73-3': {'73-4': {'73-5': {'73-6': {'73-7': {'73-8': None}}}}},
                             '94-3': {'94-4': {'94-5': {'94-6': {'94-7': {'94-8': None}}}}}
                             },
                    '64-2': {'64-3': {'64-4': {'64-5': {'64-6': {'64-7': {'64-8': None}}}}}},
                    '65-2': {'65-3': {'65-4': {'65-5': {'65-6': {'65-7': {'65-8': None}}}}},
                             '96-3': {'96-4': {'96-5': {'96-6': {'96-7': {'96-8': None}}}}},
                             },
                    '66-2': {'66-3': {'66-4': {'66-5': {'66-6': {'66-7': {'66-8': None}}}}}},
                    '68-2': {'68-3': {'68-4': {'68-5': {'68-6': {'68-7': {'68-8': None}}}}}},
                    '69-2': {'69-3': {'69-4': {'69-5': {'69-6': {'69-7': {'69-8': None}}}}}},
                    '7-2': {'7-3': {'7-4': {'7-5': {'7-6': {'7-7': {'7-8': None}}}}}},
                    '72-2': {'72-3': {'72-4': {'72-5': {'72-6': {'72-7': {'72-8': None}}}}}},
                    '75-2': {'75-3': {'75-4': {'75-5': {'75-6': {'75-7': {'75-8': None}}}}}},
                    '76-2': {'76-3': {'76-4': {'76-5': {'76-6': {'76-7': {'76-8': None}}}}}},
                    '80-2': {'80-3': {'80-4': {'80-5': {'80-6': {'80-7': {'80-8': None}}}}}},
                    '82-2': { '26-3': {'26-4': {'26-5': {'26-6': {'26-7': {'26-8': None}}}}},
                        '82-3': {'82-4': {'82-5': {'82-6': {'82-7': {'82-8': None}}}}}},

                    '9-2': {'1-3': {'1-4': {'1-5': {'1-6': {'1-7': {'1-8': None}}}}},
                        '21-3': {'21-4': {'21-5': {'21-6': {'21-7': {'21-8': None}}}}},
                            '22-3': {'22-4': {'22-5': {'22-6': {'22-7': {'22-8': None}}}}},
                            '4-3': {'4-4': {'4-5': {'4-6': {'4-7': {'4-8': None}}}}},
                            '46-3': {'46-4': {'46-5': {'46-6': {'46-7': {'46-8': None}}}}},
                            '59-3': {'59-4': {'59-5': {'59-6': {'59-7': {'59-8': None}}}}},
                            '77-3': {'77-4': {'77-5': {'77-6': {'77-7': {'77-8': None}}}}},
                            '81-3': {'81-4': {'81-5': {'81-6': {'81-7': {'81-8': None}}}}},
                            '83-3': {'83-4': {'83-5': {'83-6': {'83-7': {'83-8': None}}}}},
                            '9-3': {'9-4': {'9-5': {'9-6': {'9-7': {'9-8': None}}}}}},
                    '90-2': {'90-3': {'90-4': {'90-5': {'90-6': {'90-7': {'90-8': None}}}}}},
                    '91-2': {'3-3': {'3-4': {'3-5': {'3-6': {'3-7': {'3-8': None}}}}},
                        '48-3': {'48-4': {'48-5': {'48-6': {'48-7': {'48-8': None}}}}},
                             '58-3': {'58-4': {'58-5': {'58-6': {'58-7': {'58-8': None}}}}},
                        '74-3': {'74-4': {'74-5': {'74-6': {'74-7': {'74-8': None}}}}},
                             '78-3': {'78-4': {'78-5': {'78-6': {'78-7': {'78-8': None}}}}},
                        '87-3': {'87-4': {'87-5': {'87-6': {'87-7': {'87-8': None}}}}},
                             '88-3': {'88-4': {'88-5': {'88-6': {'88-7': {'88-8': None}}}}},
                        '91-3': {'91-4': {'91-5': {'91-6': {'91-7': {'91-8': None}}}}}},
                    '92-2': {'14-3': {'14-4': {'14-5': {'14-6': {'14-7': {'14-8': None}}}}},
                             '38-3': {'38-4': {'38-5': {'38-6': {'38-7': {'38-8': None}}}}},
                             '92-3': {'92-4': {'92-5': {'92-6': {'92-7': {'92-8': None}}}}}},
                    '93-2': {'93-3': {'93-4': {'93-5': {'93-6': {'93-7': {'93-8': None}}}}}},
                    '98-2': {'98-3': {'98-4': {'98-5': {'98-6': {'98-7': {'98-8': None}}}}}},

                }}

    atree = Individual(60, '100t',nee1=True,kind=100)
    atree.tree_dict=tree_dict
    atree.print_dict_(atree.tree_dict)
    atree.paths = atree.dg(tree_dict)
    print(atree.paths)
    atree.eval_paths(atree.indi_no)
    print(atree.acc)
    #test_population()