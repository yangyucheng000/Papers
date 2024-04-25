import pandas as pd
import ast

def process_data(txt_file):
    # 读取文本文件，并按行拆分
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 创建空的数据框
    data = {'sentence': [], 'aspect label': [], 'opinion label': [], 'sentiment label': []}

    # 遍历每一行
    for line in lines:
        # 拆分句子和注释
        try:
            sentence, annotation = line.strip().split('####')
        except ValueError:
            print(line)

        # 解析注释中的三元组
        
        annotation_mul = eval(annotation)
        # annotation = annotation[0]
        # print(type(annotation))

        # 获取三元组信息
        # import pdb; pdb.set_trace()
        aspect_label = [0] * len(sentence.split())
        opinion_label = [0] * len(sentence.split())
        for annotation in annotation_mul:
            aspect_indices, opinion_indices, sentiment = annotation
            # print(aspect_indices)
            # 生成 aspect 和 opinion 标签
            # aspect_label = [0] * len(sentence.split())
            # opinion_label = [0] * len(sentence.split())

            for idx in aspect_indices:
                aspect_label[idx] = 1
                for i in range(1, len(aspect_indices)):
                    aspect_label[aspect_indices[i]] = 2

            for idx in opinion_indices:
                opinion_label[idx] = 1
                for i in range(1, len(opinion_indices)):
                    opinion_label[opinion_indices[i]] = 2

            # 生成 sentiment 标签
            sentiment_label = [[0] * len(sentence.split()) for _ in range(len(sentence.split()))]

            if sentiment == 'NEG' or sentiment == 'neg':
                for i in range(len(aspect_indices)):
                    for j in range(len(opinion_indices)):
                        sentiment_label[aspect_indices[i]][opinion_indices[j]] = 2
            
            if sentiment == 'POS' or sentiment == 'pos':
                for i in range(len(aspect_indices)):
                    for j in range(len(opinion_indices)):
                        sentiment_label[aspect_indices[i]][opinion_indices[j]] = 1
            
            if sentiment == 'NEU' or sentiment == 'neu':
                for i in range(len(aspect_indices)):
                    for j in range(len(opinion_indices)):
                        sentiment_label[aspect_indices[i]][opinion_indices[j]] = 0

        # 将数据添加到数据框
        data['sentence'].append(sentence)
        data['aspect label'].append(aspect_label)
        data['opinion label'].append(opinion_label)
        data['sentiment label'].append(sentiment_label)

    # 创建数据框
    df = pd.DataFrame(data)

    return df

# 处理数据并保存到 TSV 文件
df = process_data('ASTE-Data-V2/res14/train_triplets.txt')
df.to_csv('ASTE-Data-V2/res14/train_triplets.tsv', sep='\t', index=False)

df = process_data('ASTE-Data-V2/res14/test_triplets.txt')
df.to_csv('ASTE-Data-V2/res14/test_triplets.tsv', sep='\t', index=False)

df = process_data('ASTE-Data-V2/res14/dev_triplets.txt')
df.to_csv('ASTE-Data-V2/res14/dev_triplets.tsv', sep='\t', index=False)
