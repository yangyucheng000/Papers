# -*- coding: utf-8 -*-
# coding=utf-8
# 将.pth文件转换为.ckpt文件
import torch
import mindspore
import mindspore as ms
print(ms.__version__)

from mindspore.train.serialization import save_checkpoint
from mindspore.dataset.text import BertTokenizer
import pickle

file_lst = ["reftext_subtest_home.pth", "reftext_subtest_oov.pth", "reftext_subtest_other.pth",
            "reftext_subtest_semantic.pth", "reftext_subtest_shelf.pth", "reftext_subtest_sport.pth",
            "reftext_subtest_street.pth", "reftext_test.pth", "reftext_train.pth", "reftext_val.pth"]
output_lst = ["reftext_subtest_home.pkl", "reftext_subtest_oov.pkl", "reftext_subtest_other.pkl",
              "reftext_subtest_semantic.pkl", "reftext_subtest_shelf.pkl", "reftext_subtest_sport.pkl",
              "reftext_subtest_street.pkl", "reftext_test.pkl", "reftext_train.pkl", "reftext_val.pkl"]
def coverct_fun():
    for j in range(len(file_lst)):
        torch_model = torch.load(file_lst[j])
        print(type(torch_model))
        print(len(torch_model))
        for i in range(5):
            print(torch_model[i])

        # 打开一个Pickle文件，如果不存在则创建
        with open(output_lst[j], "wb") as f:
            # 将列表保存为Pickle文件
            pickle.dump(torch_model, f)

def test_berttok():

    tokenizer = BertTokenizer(vocab_file="vocab.txt")
    tokens = tokenizer.encode("I have a new GPU!")
    print(tokens)

# import mindspore.dataset as ds
# import mindspore.dataset.text as text
# from mindspore.dataset.text import NormalizeForm
#
# text_file_list = ["./vocab.txt"]
# text_file_dataset = ds.TextFileDataset(dataset_files=text_file_list)
#
# # 1) If with_offsets=False, default output one column {["text", dtype=str]}
# vocab_list = ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低",
#               "思", "故", "乡","繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak",
#               "make", "small", "mistake", "##s", "during", "work", "##ing", "hour", "😀", "😃",
#               "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I", "[CLS]", "[SEP]",
#               "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]
# vocab = text.Vocab.from_list(vocab_list)
# tokenizer_op = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100,
#                                   unknown_token='[UNK]', lower_case=False, keep_whitespace=False,
#                                   normalization_form=NormalizeForm.NONE, preserve_unused_token=True,
#                                   with_offsets=False)
# text_file_dataset = text_file_dataset.map(operations=tokenizer_op)
# print(text_file_dataset)
#
# for data in text_file_dataset.create_dict_iterator():
#     print(type(data))
#     print(data["text"])



# 导入transformers库和BertTokenizer类
# from torchvision.transforms import Compose, ToTensor, Normalize
# input_transform = Compose([
#         ToTensor(),
#         Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225])
#     ])
# print(input_transform)

from mindspore import Tensor
from mindspore.ops import operations as P

x = Tensor([1, 2, 3, 4, 5])
y = P.clip_by_value(x, 2, 4)

print(y)




