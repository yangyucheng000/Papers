import pandas as pd
from mindnlp.transformers import BertTokenizer

def process_data(txt_file, tokenizer):
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

        # 解析注释中的多个三元组
        annotations = eval(annotation)

        # 处理每个三元组
        for annotation in annotations:
            aspect_indices, opinion_indices, sentiment = annotation

            # 使用分词器对句子进行分词
            tokens = tokenizer.tokenize(tokenizer.encode(sentence))

            # 初始化 aspect 和 opinion 标签
            aspect_label = [0] * len(tokens)
            opinion_label = [0] * len(tokens)

            # 对每个 aspect 和 opinion 的起始位置进行标注
            for idx in aspect_indices[0]:
                start = tokenizer.encode(sentence[:idx], add_special_tokens=False).index(1)  # 1 表示起始位置
                aspect_label[start:start + len(aspect_indices)] = [1] + [2] * (len(aspect_indices) - 1)

            for idx in opinion_indices[0]:
                start = tokenizer.encode(sentence[:idx], add_special_tokens=False).index(1)
                opinion_label[start:start + len(opinion_indices)] = [1] + [2] * (len(opinion_indices) - 1)

            # 生成 sentiment 标签
            sentiment_label = [[0] * len(tokens) for _ in range(len(tokens))]

            if sentiment.lower() == 'neg':
                for i in range(len(aspect_indices[0])):
                    for j in range(len(opinion_indices[0])):
                        start_aspect = tokenizer.encode(sentence[:aspect_indices[0][i]], add_special_tokens=False).index(1)
                        start_opinion = tokenizer.encode(sentence[:opinion_indices[0][j]], add_special_tokens=False).index(1)
                        sentiment_label[start_aspect][start_opinion] = 2
            
            if sentiment.lower() == 'pos':
                for i in range(len(aspect_indices[0])):
                    for j in range(len(opinion_indices[0])):
                        start_aspect = tokenizer.encode(sentence[:aspect_indices[0][i]], add_special_tokens=False).index(1)
                        start_opinion = tokenizer.encode(sentence[:opinion_indices[0][j]], add_special_tokens=False).index(1)
                        sentiment_label[start_aspect][start_opinion] = 1
            
            if sentiment.lower() == 'neu':
                for i in range(len(aspect_indices[0])):
                    for j in range(len(opinion_indices[0])):
                        start_aspect = tokenizer.encode(sentence[:aspect_indices[0][i]], add_special_tokens=False).index(1)
                        start_opinion = tokenizer.encode(sentence[:opinion_indices[0][j]], add_special_tokens=False).index(1)
                        sentiment_label[start_aspect][start_opinion] = 0

            # 将数据添加到数据框
            data['sentence'].append(sentence)
            data['aspect label'].append(aspect_label)
            data['opinion label'].append(opinion_label)
            data['sentiment label'].append(sentiment_label)

    # 创建数据框
    df = pd.DataFrame(data)

    return df

# 使用 BERT 分词器作为示例
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df_train = process_data('ASTE-Data-V2/res14/train_triplets.txt', tokenizer)
df_train.to_csv('ASTE-Data-V2/res14/train_triplets.tsv', sep='\t', index=False)

df_test = process_data('ASTE-Data-V2/res14/test_triplets.txt', tokenizer)
df_test.to_csv('ASTE-Data-V2/res14/test_triplets.tsv', sep='\t', index=False)

df_dev = process_data('ASTE-Data-V2/res14/dev_triplets.txt', tokenizer)
df_dev.to_csv('ASTE-Data-V2/res14/dev_triplets.tsv', sep='\t', index=False)
