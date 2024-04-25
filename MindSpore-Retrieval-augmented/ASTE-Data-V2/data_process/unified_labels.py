import pdb

class TagReader():
    # 0: neu, pos: 1, neg: 2
    @classmethod
    def read_init_data_V2(cls, file):
        f = open(file, 'r', encoding = 'utf-8')
        for line in f:
            line = line.strip().split('####')
            inputs = line[0].split() # sentence
            # a_output = line[1].split() # aspect terms
            # o_output = line[2].split() # opinion terms
            raw_triplets = eval(line[1]) # triplets
            pdb.set_trace()
            # prepare tagging sequence
            # output = ['O' for x in range(len(inputs))]
            # polarity = [0 for x in range(len(inputs))]
            # for i, t in enumerate(t_output):
            #     t = t.split('=')[1]
            #     if t != 'O':
            #         output[i] = t
    @staticmethod
    def raw2span(raw_triplets):
        '''
        [([0, 1, 2], [4], 'POS'), ([0, 1, 2], [7], 'POS')] -> []
        [([4, 5], [2], 'POS'), ([11, 12, 13, 14, 15], [2], 'POS')] -> []
        '''
        for raw in raw_triplets:
            aspect_start, aspect_end = raw[0][0], raw[0][-1]
            opinion_start, opinion_end = raw[1][0], aw[1][-1]

if __name__ == '__main__':
    file = '../14res/train_triplets.txt'
    TagReader.read_init_data_V2(file)