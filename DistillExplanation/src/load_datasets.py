import mindspore
import pandas as pd
from mindspore.dataset import GeneratorDataset, transforms

from mindnlp.transformers import T5Tokenizer
import numpy as np

# prepare dataset

class COPADataset:
    """WinograndeDataset Dataset"""

    def __init__(self, path):
        self.path = path
        self._questions, self._choice_a, self._choice_b,  self._answers, self._results, self._explanations = [], [], [], [], [], []
        self._load()

    def _load(self):
        _df = pd.read_csv(self.path)
        for i,row in _df.iterrows():

            self._questions.append(row['prompt'])
            self._choice_a.append(row['a1'])
            self._choice_b.append(row['a2'])
            self._answers.append(row['answer'])

            self._results.append(row['result'])
            self._explanations.append(row['explanation'])


    def __getitem__(self, index):
        return (self._questions[index], 
                self._choice_a[index],self._choice_b[index], 
                self._answers[index], self._results[index], 
                self._explanations[index])

    def __len__(self):
        return len(self._questions)
    
def process_copa_dataset(source, tokenizer, on_gound_truth_label=True, max_seq_len=256, batch_size=32, shuffle=True, test=False):
    is_ascend = mindspore.get_context('device_target') == 'GPU'
    
    column_names = ['question', 'A', 'B', 'answer','result', 'explanation']
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int64)
    
    def tokenize_and_pad(
        question, choiceA, choiceB,
        answer_choice, result_choice,
        _expl 
    ):

        if on_gound_truth_label:
            answer = choiceA if answer_choice == 'A' else choiceB
        else: 
            answer = choiceA if result_choice == 'A' else choiceB
        
        input_string = f"{question}" 
        answer_string = f"{answer} explanation: {_expl}"

        tokenized_input = tokenizer(input_string, padding='max_length', truncation=True, max_length=max_seq_len)
        tokenized_answer = tokenizer(answer_string, padding='max_length', truncation=True, max_length=max_seq_len)

        label_ids = tokenized_answer['input_ids']
        label_ids = label_ids + (1 - np.array(tokenized_answer['attention_mask'])) * -100
        

        
        return tokenized_input['input_ids'], tokenized_input['attention_mask'], \
                label_ids, tokenized_answer['attention_mask'], \
                    tokenized_input['input_ids']
                
    # map dataset
    dataset = dataset.map(
        operations=tokenize_and_pad, 
        input_columns=['question', 'A', 'B', 'answer','result', 'explanation'], 
        output_columns=['input_ids', 'attention_mask','labels', 'decoder_attention_mask', 'question_encoding']
    )
    
    #dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask', 'question_encoding'])
    
    dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['attention_mask'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['labels'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['decoder_attention_mask'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['question_encoding'])
    
    
    # batch dataset
    if is_ascend:
        #dataset = dataset
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                         'attention_mask': (None, 0)})

    return dataset


def process_score_model_copa_dataset(source, tokenizer, on_gound_truth_label=True, max_seq_len=256, batch_size=8, shuffle=True, test=False):
    
    column_names = ['question', 'A', 'B', 'answer','result', 'explanation']
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int64)
    
    def read_text(
        question, choiceA, choiceB,
        answer_choice, result_choice,
        _expl 
    ):

        answer = choiceA if result_choice == 'A' else choiceB
        
        input_string = f"{_expl}" 
        answer_string = f"{answer}"

        return input_string, answer_string

    def tokenize_and_pad(
        input_string, answer_string
    ):
        _tokenized = tokenizer(text=input_string, padding='max_length', truncation=True, max_length=max_seq_len)
        _decodings = tokenizer(text=answer_string, padding='max_length', truncation=True, max_length=max_seq_len)
#        import pdb; pdb.set_trace()
        pred_labels = list(_decodings['input_ids'] + (-100) * (1 - np.array(_decodings['attention_mask'])))
  

        return _tokenized['input_ids'], _tokenized['attention_mask'], pred_labels, _decodings['attention_mask']

                
    # map dataset
    dataset = dataset.map(
        operations=read_text,
        input_columns=['question', 'A', 'B', 'answer','result', 'explanation'],
        output_columns=['input_string', 'answer_string']
    )
    
    dataset = dataset.map(
        operations=tokenize_and_pad, 
        input_columns=['input_string', 'answer_string'], 
        output_columns=['input_ids', 'attention_mask','labels', 'labels_attention_mask']
    )
    
    
    dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['attention_mask'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['labels'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['labels_attention_mask'])

    
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def process_multi_task_copa_dataset(source, tokenizer, on_gound_truth_label=True, max_seq_len=512, batch_size=8, shuffle=True, test=False):
    is_ascend = mindspore.get_context('device_target') == 'GPU'
    
    column_names = ['question', 'A', 'B', 'answer','result', 'explanation']
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int64)
    
    def read_text(
        question, choiceA, choiceB,
        answer_choice, result_choice,
        _expl 
    ):

        if on_gound_truth_label:
            answer = choiceA if answer_choice == 'A' else choiceB
        else: 
            answer = choiceA if result_choice == 'A' else choiceB
        
        input_string = f"{question}"
        input_string = input_string.split('\n')
        input_string = ' '.join(input_string)

        label_input_string = f"[label] {input_string}" 
        rationale_input_string = f"[rationale] {input_string}"
        
        label_answer = f"{answer}"
        rationale_answer = f"{_expl}"
        
        return label_input_string, label_answer, rationale_input_string, rationale_answer



    def tokenize_and_pad(
        label_input_string, 
        label_answer, 
        rationale_input_string, 
        rationale_answer
    ):
        pred_tokenized = tokenizer(text=label_input_string, padding='max_length', truncation=True, max_length=max_seq_len)
        pred_decodings = tokenizer(text=label_answer, padding='max_length', truncation=True, max_length=32)
        pred_labels = list(pred_decodings['input_ids'] + (-100) * (1 - np.array(pred_decodings['attention_mask'])))
        
        
        rationale_tokenized = tokenizer(text=rationale_input_string, padding='max_length', truncation=True, max_length=max_seq_len)
        rationale_decodings = tokenizer(text=rationale_answer, padding='max_length', truncation=True, max_length=max_seq_len)
        rationale_labels = list(rationale_decodings['input_ids'] + (-100) * (1 - np.array(rationale_decodings['attention_mask'])))
        
        return pred_tokenized['input_ids'], pred_tokenized['attention_mask'], pred_labels, pred_decodings['attention_mask'], \
            rationale_tokenized['input_ids'], rationale_tokenized['attention_mask'], rationale_labels, rationale_decodings['attention_mask']
    

                
    # map dataset
    col_names = ['pred_input_ids', 'pred_attention_mask','pred_labels', 'pred_labels_attention_mask', \
                 'rationale_input_ids', 'rationale_attention_mask','rationale_labels', 'rationale_labels_attention_mask',]
    
    
    dataset = dataset.map(
        operations=read_text, 
        input_columns=['question', 'A', 'B', 'answer','result', 'explanation'], 
        output_columns=['label_input_string', 'label_answer', 'rationale_input_string', 'rationale_answer']
    )
    
    
    dataset = dataset.map(
        operations=tokenize_and_pad , 
        input_columns=['label_input_string', 'label_answer', 'rationale_input_string', 'rationale_answer'], 
        output_columns=col_names 
    )

    for col in  col_names:
        dataset = dataset.map(operations=type_cast_op, input_columns=[col])

    
    
    # batch dataset
    if is_ascend:
        #dataset = dataset
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                         'attention_mask': (None, 0)})

    return dataset


def process_dataset(source, tokenizer, on_gound_truth_label=True, max_seq_len=256, batch_size=8, shuffle=True):
    is_ascend = mindspore.get_context('device_target') == 'GPU'
    
    column_names = ['qID', 'Sentence', 'A', 'B', 'Answer', 'prompt', 'Result', 'Explanation', 'isCorrect']
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int32)
    def tokenize_and_pad(
        question, choiceA, choiceB,
        answer_choice, 
        result, 
        _expl 
    ):

        if on_gound_truth_label:
            answer = choiceA if answer_choice == 'A' else choiceB
        else: 
            answer = choiceA if result == 'A' else choiceB
        
        input_string = f"fill in the blank and explain your reason for the commensense inference question: {question} choice: {choiceA} choice: {choiceB}" 
        answer_string = f"{answer} explanation: {_expl}"

        tokenized_input = tokenizer(input_string, padding='max_length', truncation=True, max_length=max_seq_len)
        tokenized_answer = tokenizer(answer_string, padding='max_length', truncation=True, max_length=max_seq_len)

        return tokenized_input['input_ids'], tokenized_input['attention_mask'], \
                tokenized_answer['input_ids'], tokenized_answer['attention_mask'], \
                    tokenized_input['input_ids']
                
    # map dataset
    dataset = dataset.map(
        operations=tokenize_and_pad, 
        input_columns=['Sentence', 'A', 'B', 'Answer', 'Result', 'Explanation'], 
        output_columns=['input_ids', 'attention_mask','labels', 'decoder_attention_mask', 'question_encoding']
    )
    
    #dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask', 'question_encoding'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['attention_mask'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['labels'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['decoder_attention_mask'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['question_encoding'])
    
    
    # batch dataset
    if is_ascend:
        #dataset = dataset
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                         'attention_mask': (None, 0)})

    return dataset


def load_t5_dataset(data_dir, multi_task=False, score_model=False): 
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    data_files = {"train": f"{data_dir}/train.csv", "validation": f"{data_dir}/dev.csv", "test": f"{data_dir}/test.csv"}

    
    if score_model: 
        train_dataset = process_score_model_copa_dataset(COPADataset(data_files['train']), tokenizer)
        valid_dataset = process_score_model_copa_dataset(COPADataset(data_files['validation']), tokenizer)
        test_dataset = process_score_model_copa_dataset(COPADataset(data_files['test']), tokenizer, shuffle=False, test=True)
        
        saved_columns = ['input_ids', 'attention_mask', 'labels']

    elif multi_task:
        train_dataset = process_multi_task_copa_dataset(COPADataset(data_files['train']), tokenizer)
        valid_dataset = process_multi_task_copa_dataset(COPADataset(data_files['validation']), tokenizer)
        test_dataset = process_multi_task_copa_dataset(COPADataset(data_files['test']), tokenizer, shuffle=False, test=True)
        
        
        saved_columns = ['pred_input_ids', 'pred_attention_mask','pred_labels', 'pred_labels_attention_mask',  \
                 'rationale_input_ids', 'rationale_attention_mask','rationale_labels', 'rationale_labels_attention_mask']
        
    else:
        train_dataset = process_copa_dataset(COPADataset(data_files['train']), tokenizer)
        valid_dataset = process_copa_dataset(COPADataset(data_files['validation']), tokenizer)
        test_dataset = process_copa_dataset(COPADataset(data_files['test']), tokenizer, shuffle=False, test=True)
        
        saved_columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask', 'question_encoding']
    
    train_dataset = train_dataset.project(columns=saved_columns)
    valid_dataset = valid_dataset.project(columns=saved_columns)
    test_dataset = test_dataset.project(columns=saved_columns)
    

    #print(next(test_dataset.create_dict_iterator()))
    return train_dataset , valid_dataset, test_dataset
 
   
    
if __name__ == "__main__":
    data_dir = 'data/copa/data'
    load_t5_dataset(data_dir, multi_task=False, score_model=True)
    
