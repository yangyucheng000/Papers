
import sys
sys.path.append('../')

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from dataPrepare.utils import *
import pandas as pd
from collections import Counter
# from torch_sparse import SparseTensor
import argparse
import os
import pickle

import torch.nn.functional as F
from torch_scatter import scatter_add

from collections import Counter


class PreTextData(object):
    """docstring for MonoTextData"""

    def __init__(self, fname, hop_num=3, min_length=10, max_length=None, vocab=None, edge_threshold=10, num_path=200):
        super(PreTextData, self).__init__()

        self.fname = fname

        self.data, self.vocab, self.dropped, self.labels, self.word_count, self.train_split, self.itemids \
            = self._read_corpus(fname, vocab, max_length=max_length, min_length=min_length)
        self.hop_num = hop_num
        self.num_path = num_path
     
        self.pairs = pickle.load(open(self.fname.replace('overall_stop.csv', 'whole_pairs_path%d.pkl' % num_path), 'rb'))
        self.doc_pairs = pickle.load(open(self.fname.replace('overall_stop.csv', 'all_doc_pairs_path%d.pkl' % num_path), 'rb'))

        self.traiples = pickle.load(open(self.fname.replace('overall_stop.csv', 'whole_triples_path%d.pkl' % num_path), 'rb'))
        self.doc_triples = pickle.load(open(self.fname.replace('overall_stop.csv', 'all_doc_triples_path%d.pkl' % num_path), 'rb'))

        self.relation_map, self.unique_nodes_mapping, self.concept_map = self.get_maps()

    def __len__(self):
        return len(self.data)

    def get_maps(self):
        relation_map = pickle.load(open(self.fname.replace('overall_stop.csv', 'relation_map_path%d.pkl' % self.num_path), 'rb'))
        unique_nodes_mapping = pickle.load(open(self.fname.replace('overall_stop.csv', 'unique_nodes_mapping_path%d.pkl' % self.num_path), 'rb'))
        concept_map = pickle.load(open(self.fname.replace('overall_stop.csv', 'concept_map_path%d.pkl' % self.num_path), 'rb'))
        return relation_map, unique_nodes_mapping, concept_map

    def _read_corpus(self, fname, vocab: VocabEntry, min_length=10, max_length=1000):
        labels = []
        data = []
        # add
        # tags = []
        tran_split = []
        itemids = []
        dropped = 0
        word_count = 0
        csvdata = pd.read_csv(fname, header=0, dtype={'label': int, 'train': int})
        for i, ss in enumerate(csvdata[['label', 'content', 'train', 'idx']].values):
            lb = ss[0]
            # regard ; as one word, which can be deleted if needed
            try:
                split_line = ss[1].split()
            except:
                print(ss[1])
            if len(split_line) < min_length:
                dropped += 1
                continue
            if max_length is not None:
                if len(split_line) > max_length:
                    dropped += 1
                    continue

            # add here 
            # doc = nlp(ss[1])
            # pos_tags = [token.tag_ for token in doc]
            idxs = [vocab[word] for word in split_line if vocab[word] > 0]  
            # tag = [pos_tags[i] for i, word in enumerate(split_line) if vocab[word] > 0]

            word_num = len(idxs)
            if word_num < 3:
                dropped += 1
                continue

            labels.append(lb)
            data.append(idxs)
            # add
            # tags.append(tag)
            itemids.append(ss[3])
            tran_split.append(int(ss[2]))
            word_count += word_num
        print('read corpus done!')
        return data, vocab, dropped, labels, word_count, tran_split, itemids


    def get_pair_dict(self, threshold):
        tmp = [tuple(t) for t in self.pairs]
        coun_dct = Counter(tmp)
        self.pair_dct = {k: coun_dct[k] for k in coun_dct if coun_dct[k] > threshold and k[0] != k[1]}
        sorted_key = sorted(self.pair_dct.keys(), key=lambda x: self.pair_dct[x], reverse=True)    # a list
        for i, key in enumerate(sorted_key):
            self.pair_dct[key] = i + 1  # start from 1
        self.whole_edge = np.array([k for k in sorted_key]).transpose() 
        self.whole_edge_w = np.array([coun_dct[k] for k in sorted_key])
        print('pairVocab done!')
        print(self.whole_edge.shape)

    def process_sent(self, sent, pairs, triples):

        triple_idx= triples2index(triples, self.relation_map, self.concept_map, self.unique_nodes_mapping) # [[id1, id2, id3], [id4, id5, id6]]
        # print(len(triple_idx))
        xg = np.array(triple_idx)  
        xg = unique_rows(xg).astype('int64')

        text = [self.vocab.id2word(idx) for idx in sent]

        gcn_input_data, gcn_idx = generate_graph(xg, text, self.concept_map, self.unique_nodes_mapping, len(self.relation_map))

        L = len(sent)
        edge_ids = []

        tmp = [tuple(t) for t in pairs]
        dct = Counter(tmp)
        keys = dct.keys()
        r, c, v = [], [], []
        for k in keys:
            try:
                edge_id = self.pair_dct[k]
            except:
                continue

            r.append(k[0])    
            c.append(k[1])    
            v.append(dct[k])  
            edge_ids.append(edge_id)    
        # edge_index = np.array([c, r]) ### 每一行为入度邻居a_{i,j} j->i
        edge_index = np.array([r, c])  ### 每一行为出度邻居a_{i,j} i->j
        edge_w = np.array(v)

        idxs = np.unique(edge_index.reshape(-1))    
        idx_w_dict = Counter(sent)

        idx_w = []
        lens = 0
        for id in idxs:
            idx_w.append(idx_w_dict[id])    
            lens += idx_w_dict[id]          

        sidxs = []    
        for id in sent:
            if id not in idxs and id not in sidxs:    
                sidxs.append(id)
                idx_w.append(idx_w_dict[id])
                lens += idx_w_dict[id]

        if len(idxs) > 0 and len(sidxs) > 0:
            all_idxs = np.hstack([idxs, sidxs])
        elif len(idxs) == 0 and len(sidxs) > 0:
            all_idxs = np.array(sidxs)
        else:
            all_idxs = idxs
        # idxs.dtype=np.int
        assert lens == len(sent)
        # if max(all_idxs)>10000:
        #     import ipdb
        #     ipdb.set_trace()
        if len(idxs) > 0:
            idxs_map = np.zeros(max(all_idxs) + 1)
            idxs_map[all_idxs] = range(len(all_idxs))
            edge_index = idxs_map[edge_index]
        else:
            edge_index = np.array([[], []])

        # get the map between all_idxs and sent
        sent_map = {token:i for i, token in enumerate(sent)}    # token -> id
        idx2sent = [sent_map[idx] for idx in all_idxs]          # token -> id in sent - align the token in sent to the token in all_idxs
        return all_idxs, idx_w, edge_index, edge_w, edge_ids, L, gcn_input_data, gcn_idx, idx2sent
   

def triples2index(triples, relation_map, concept_map, unique_nodes_mapping):
    '''
    transform the concepts and relations in each triple to their corresponding ids in the map
    '''

    triple_idx = []

    for triple in triples:  
        try:
            srcMap = concept_map[triple[0]]
            relMap = relation_map[triple[1]]
            distMap = concept_map[triple[2]]
           
            srcMap, distMap = unique_nodes_mapping[srcMap], unique_nodes_mapping[distMap]
        except:
            continue
        triple = [srcMap, relMap, distMap]
        triple_idx.append(triple)

    if len(triple_idx) == 0:   # 如果输入的triples为空，对应句子中没有triples的情况, 使用特殊的triple得到对应的id
        None_id = concept_map["[NONE]"]
        None_id = unique_nodes_mapping[None_id]
        rel_id = relation_map["Synonym"]
        triple_idx.append([None_id, rel_id, None_id])

    return triple_idx

def generate_graph(triplets, tokens, concept_map, unique_nodes_mapping, num_rels):
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    src = torch.tensor(src, dtype=torch.long).contiguous()
    dst = torch.tensor(dst, dtype=torch.long).contiguous()
    rel = torch.tensor(rel, dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    NONE_ID = concept_map["[NONE]"]
    # PAD_ID = concept_map["[PAD]"]
    NONE_ID = unique_nodes_mapping[NONE_ID]
    # PAD_ID = unique_nodes_mapping[PAD_ID]

    # 存放当前这个graph中的所有节点
    uniq_entity = list(uniq_entity)    
    # uniq_entity.extend([NONE_ID, PAD_ID])
    uniq_entity.extend([NONE_ID])

    # 将句子/文档中的单词映射为concept节点
    token_ids = []
    for idx in range(len(tokens)):  # idx_map =[0, 1, 1, 2]
        token = tokens[idx]
        if token in concept_map.keys():
            concept_id = concept_map[token]
        else:
            token_ids.append(NONE_ID)
            continue
        if concept_id in unique_nodes_mapping.keys():
            token_ids.append(unique_nodes_mapping[concept_id])
        else:
            token_ids.append(NONE_ID)

    # 往uniq_entity中添加补充的节点
    for token in token_ids:
        if token not in uniq_entity:
            uniq_entity.append(token)

    uni_map = {token: idx for idx, token in enumerate(uniq_entity)}
    idx_arr = []
 
    # 存放当前句子/文档中每个单词在uniq_entity中的编号
    # 后续方便得到每个句子/文档中每个单词的embedding(对应单词在句子/文档中的原始出现顺序)
    for token in token_ids:
        idx_arr.append(uni_map[token])    

    #
    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.array(uniq_entity))    
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data, idx_arr


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def unique_rows(a):
    """
    Drops duplicate rows from a numpy 2d array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


class MyData(Data):
    def __init__(self, x=None, edge_w=None, edge_index=None, x_w=None, edge_id=None, y=None):
        super(MyData, self).__init__()
        if x is not None:
            self.x = x
        if edge_w is not None:
            self.edge_w = edge_w
        if edge_index is not None:
            self.edge_index = edge_index
        if x_w is not None:
            self.x_w = x_w
        if edge_id is not None:
            self.edge_id = edge_id
        if y is not None:
            self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        elif key == 'x':
            return 0
        elif key == 'edge_id':
            return 0
        else:
            return 0


class GraphDataset(InMemoryDataset):
    def __init__(self, root, hop_num=1, vocab=None, transform=None, pre_transform=None, STOPWORD=False,
                 edge_threshold=10, num_path=200):
        self.rootPath = root
        self.stop_str = '_stop' if STOPWORD else ''
        self.edge_threshold = edge_threshold
        self.hop_num = hop_num
        self.num_path = num_path
        if vocab is None:    
            self.vocab = VocabEntry.from_corpus(os.path.join(self.rootPath, 'vocab%s.txt_2' % self.stop_str), withpad=False)
        else:
            self.vocab = vocab

        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.whole_edge, \
        self.word_count, self.dropped, self.whole_edge_w, self.all_gcn_inputs, self.all_gcn_idxs, self.all_idx2sent, self.ori_data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):    
        return []

    @property
    def processed_file_names(self):    
        return [self.rootPath + '/rgcn_graph_hop%d_dataset%s_edgethres%d_path%d.pt' % (self.hop_num, self.stop_str, self.edge_threshold, self.num_path)]

    def download(self):
        pass

    # 需要修改这里
    def process(self):
        dataset = PreTextData(self.rootPath + '/overall%s.csv' % self.stop_str, hop_num=self.hop_num,
                              vocab=self.vocab, min_length=5, max_length=None,
                              edge_threshold=self.edge_threshold, num_path=self.num_path)  # TODO important parameter for different datasets
        
        dataset.get_pair_dict(self.edge_threshold)
        data_list = []
        used_list = []

        all_gcn_inputs = []
        all_gcn_idxs = []

        all_idx2sent = []
        for i in range(len(dataset)):
            # for i in range(10):
            sent = dataset.data[i]
            label = dataset.labels[i]
            train = dataset.train_split[i]
            pairs = dataset.doc_pairs[i]
            triples = dataset.doc_triples[i]
            # idxs, idx_w, edge_index, edge_w, edge_id, L = dataset.process_sent(sent, tag)
            idxs, idx_w, edge_index, edge_w, edge_id, L, gcn_input, gcn_idx, idx2sent = dataset.process_sent(sent, pairs, triples)

            if edge_index.shape[1] >= 0:   
                used_list.append(dataset.itemids[i])
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                x = torch.tensor(idxs, dtype=torch.long)
                edge_w = torch.tensor(edge_w, dtype=torch.float)
                y = torch.tensor(label, dtype=torch.long).unsqueeze(0)
                train = torch.tensor(train, dtype=torch.long).unsqueeze(0)
                idx_w = torch.tensor(idx_w, dtype=torch.float)
                edge_id = torch.tensor(edge_id, dtype=torch.long)
                d = MyData(x=x, edge_w=edge_w, edge_index=edge_index,
                           x_w=idx_w, edge_id=edge_id, y=y)
                d.train = train
                # d.gram = gram
                d.graphy = y
                # d.gramlength = L
                data_list.append(d)

            all_gcn_inputs.append(gcn_input)
            all_gcn_idxs.append(gcn_idx)
            all_idx2sent.append(idx2sent)

        # print(len(all_gcn_inputs))
        # print(len(data_list))

        np.save(self.rootPath + '/used_list_path%d_edgethreshold%d' % (self.num_path, self.edge_threshold), used_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices, dataset.whole_edge, dataset.word_count, dataset.dropped, dataset.whole_edge_w, all_gcn_inputs, all_gcn_idxs, all_idx2sent, dataset.data), self.processed_paths[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='reuters', help='the dataset to be processed.')
    parser.add_argument('--edge_threshold', type=int, default=2, help='the threshold to be used to clean edge.')
    parser.add_argument('--hop_num', type=int, default=2, help='the hop number to obtain the relationship between two concepts')
    parser.add_argument('--num_path', type=int, default=100, help='the number of shortest path')

    args = parser.parse_args()

    from settings import *
    from dataPrepare.graph_data_concept_v4 import GraphDataset

    if args.dataset == '20news':
        if args.hop_num == 1:
            root_path = NEWS20_ADDR
        elif args.hop_num == 2:
            root_path = NEWS20_ADDR_2
        elif args.hop_num == 3:
            root_path = NEWS20_ADDR_3
    elif args.dataset == 'tmn':
        if args.hop_num == 1:
            root_path = TMN_ADDR
        elif args.hop_num == 2:
            root_path = TMN_ADDR_2
        elif args.hop_num == 3:
            root_path = TMN_ADDR_3
    elif args.dataset == 'tmn_all_content':
        if args.hop_num == 1:
            root_path = TMN_ALL_CONTENT_ADDR    
    elif args.dataset == 'reuters':
        if args.hop_num == 1:
            root_path = Reuters_ADDR
        elif args.hop_num == 2:
            root_path = Reuters_ADDR_2
        elif args.hop_num == 3:
            root_path = Reuters_ADDR_3
    elif args.dataset == 'wos11967':
        if args.hop_num == 1:
            root_path = WOS_ADDR
        elif args.hop_num == 2:
            root_path = WOS_ADDR_2
        elif args.hop_num == 3:
            root_path = WOS_ADDR_3

    data = GraphDataset(root=root_path, STOPWORD=True, hop_num=args.hop_num, edge_threshold=args.edge_threshold, num_path=args.num_path)
    print('hop_num: ', args.hop_num)
    print('edge_threshold: ', args.edge_threshold)
    print('num_path', args.num_path)
    print('docs', len(data))
    print('the whole edge set', data.whole_edge.shape)
    print('data split',Counter(data.data.train.cpu().numpy()))
    print('tokens', data.data.x_w.sum().item())
    print('edges', data.data.edge_w.sum().item())
    print('vocab', len(data.vocab))

    

    """
    WOS
    hop_1:
    path50: 0 - 48191 1 - 31596
    path100: 0 - 50179 1 - 33520

    hop_2:
    path50: 0 - 206940 1 - 83524 2 - 50017 3 - 34747
    path100: 0 - 386002 1 - 162931  2 - 99157 3 - 69696 4 - 52549 5 - 41782 6 - 34227

    20news
    hop_1:
    path50: 0 - 47231 1 - 28922
    path100: 0 - 54256 1 - 35226

    hop_2:
    path50: 0 - 332396 1 - 107639 2 - 55488 3 - 34459
    path100: 0 - 518893 1 - 186446 2 - 100467 3 - 64172 4 - 45245 5 - 33816

    reuters
    hop_1:
    path50: 0 - 19382 1 - 11867
    path100: 0 - 22325 1 - 14283

    hop_2:
    path50: 0 - 121039 1 - 47107 2 - 26703 3 - 18137 4 - 13396
    path100: 0 - 196366 1 - 80870 2 - 46818 3 - 31839 4 - 23588 5 - 18381 6 - 14809
    """

