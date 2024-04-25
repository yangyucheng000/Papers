from utils import Utils, GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep

class FitnessEvaluate(object):

    def __init__(self, individuals,kind):
        self.individuals = individuals
        self.kind=kind
        #self.log = log

    def generate_to_python_file(self):
        print('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        print('Finish the generation of python files')

    def evaluate(self):
        """
        load fitness from cache file
        """
        print('Query fitness from cache')
        #map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            # _key, _str = indi.uuid()
            # if _key in _map:
            #     _count += 1
            #     _acc = _map[_key]
            #print(indi.paths)
            print('Hit the cache for %s, acc:%.5f'%(indi.indi_no, indi.acc))
        #     indi.acc = float(_acc)
        # self.log.info('Total hit %d individuals for fitness'%(_count))

        has_evaluated_offspring = False
        for indi in self.individuals:
            if indi.acc < 0:
                # p = Process(target=indi.eval_paths, args=())
                # p.start()
                has_evaluated_offspring = True
                #print(indi.tree_dict)
                print(indi.paths)
                indi.eval_paths(indi.indi_no)
                print('%s 的acc: %.5f,fitvaule:%.5f ' % (indi.indi_no, indi.acc,indi.fitvalue))
            else:
                file_name = indi.indi_no
                print('%s has inherited the fitness as %.5f, no need to evaluate'%(file_name, indi.acc))
                f = open('./populations_%2d/after_%s.txt'%(self.kind,file_name[4:6]), 'a+')
                f.write('%s=%.5f\n'%(file_name, indi.acc))
                f.flush()
                f.close()



        if has_evaluated_offspring:
            file_name = './populations_%2d/after_%s.txt'%(self.kind,self.individuals[0].indi_no[4:6])
            if os.path.exists(file_name):
                f = open(file_name, 'r')
                fitness_map = {}
                for line in f:
                    if len(line.strip()) > 0:
                        line = line.strip().split('=')
                        fitness_map[line[0]] = float(line[1])
                f.close()
                for indi in self.individuals:
                    if indi.acc == -1:
                        if indi.id not in fitness_map:
                            print('The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds'%(indi.indi_no, file_name))
                            sleep(120) #
                        indi.acc = fitness_map[indi.indi_no]
        else:
            print('None offspring has been evaluated')

       # Utils.save_fitness_to_cache(self.individuals)