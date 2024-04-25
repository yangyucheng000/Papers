from utils import StatusUpdateTool, Utils, Log
from population import Population
from evaluate import FitnessEvaluate
from Crossover import CrossoverAndMutation
from selection import Selection
import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
import os
import shutil
#python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：
def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data +'/' + i#当前文件夹的下面的所有东西的绝对路径
        #print(file_data )
        if os.path.isfile(file_data) == True and file_data[-1]!='t' and file_data[-1]!='g':#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            if file_data[-1]!='t' and file_data[-1]!='g':
                del_file(file_data)
                os.rmdir(file_data)



class EvolveCNN(object):
    def __init__(self, params,kind):
        self.params = params
        self.pops = None
        self.parent_pops = None
        self.max_acc=[]
        self.avg_acc = []
        self.kind=kind

    def initialize_population(self):
        #StatusUpdateTool.begin_evolution()
        pops = Population(self.params ,0,self.kind)
        #print(pops)
        pops.initialize()
        self.pops = pops
        #Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals,self.kind)
        #fitness.generate_to_python_file()
        fitness.evaluate()


    def crossover_and_mutation(self,curgen):
        #print(self.pops.individuals)
        cm = CrossoverAndMutation(1, 0.1, self.pops.individuals, _params={'gen_no': self.params['gen_no']})
        offspring = cm.process(curgen)
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        v_list = []
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.fitvalue)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.fitvalue)

        _str = []
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'Indi-%s-%.5f-%.5f'%(indi.indi_no, indi.acc,indi.fitvalue)
            _str.append(_t_str)
        for _, indi in enumerate(self.parent_pops.individuals):
            _t_str = 'Pare-%s-%.5f%.5f'%(indi.indi_no, indi.acc,indi.fitvalue)
            _str.append(_t_str)


        #add log
        # find the largest one's index
        max_index = np.argmax(v_list)
        selection = Selection()
        selected_index_list = selection.Truncation_selection(v_list, k=60)

        if max_index not in selected_index_list:
            first_selectd_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selectd_v_list)
            selected_index_list[min_idx] = max_index

        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id(代数和id)"""
        next_gen_pops = Population(100, self.pops.gen_no+1,self.kind)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
        acc_gen=[]

        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%.5f'%(indi.indi_no, indi.acc,indi.fitvalue)
            acc_gen.append(indi.acc)
            print(_t_str)
            _str.append(_t_str)
        _file = './populations_%2d/ENVI_%2d.txt'%(self.kind,self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)
        self.max_acc.append(max(acc_gen))
        self.avg_acc.append(sum(acc_gen)/len(acc_gen))
        #_file = './populations/acc_%2d.txt' % (self.pops.gen_no)
        #Utils.write_to_file('new -%s-%.5f-%.5f'%(indi.indi_no, indi.acc,indi.fitvalue), _file)
        #Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)
        _str1=[]
        if self.params['gen_no']==29:
            for _, indi in enumerate(self.pops.individuals):
                _t_str1 = 'new -%s-%.5f-%.5f\n%s\n%s' % (indi.indi_no, indi.acc,indi.fitvalue,indi.paths,indi.tree_dict)
                print(_t_str1)
                indi.print_dict_(indi.tree_dict)
                _str1.append(_t_str1)
            _file = './populations_%2d/lastENVI_%2d.txt' % (self.kind,self.pops.gen_no)
            Utils.write_to_file('\n'.join(_str1), _file)

    def do_work(self, max_gen):
        #Log.info('*'*25)
        # the step 1
        gen_no = 0
        if gen_no!=0:
            print('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            print('new_gen:',gen_no)
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            print('Initialize...')
            self.initialize_population()
        print('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        print('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1

        #self.initialize_population()
        for curr_gen in range(gen_no, max_gen):
            if curr_gen<max_gen-1:
                del_file('./populations_%2d/'%(self.kind))
            self.params['gen_no'] = curr_gen
            #step 3
            print('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.crossover_and_mutation(curr_gen)
            print('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            print('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            print('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))

            self.environment_selection()
            print('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))

            # 画图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
            maxacc = np.array(self.max_acc)
            t = range(maxacc.size)
            ax1.plot(t, maxacc, color='r', linestyle="-", marker='*', label='True')

            # plt.title('block {}-SIC Prediction by Grey Model (GM(1,1))'.format(j))
            # ax1.savefig('./populations/maxacc.png')

            avgacc = np.array(self.avg_acc)
            t1 = range(avgacc.size)
            ax2.plot(t1, avgacc, color='g', linestyle="-", marker='*', label='True')

            # plt.title('block {}-SIC Prediction by Grey Model (GM(1,1))'.format(j))
            plt.show()
            plt.savefig('./populations_%2d/%2d_acc.png'%(self.kind,curr_gen))



if __name__ == '__main__':
    params = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', '-k', default=100,type=int)
    args = parser.parse_args()
    kind=args.kind
    print(kind)
    del_file('./populations_%2d'%(kind))
    evoCNN = EvolveCNN(params,kind)
    evoCNN.do_work(max_gen=30)
    print(evoCNN.max_acc)
    print(evoCNN.avg_acc)

