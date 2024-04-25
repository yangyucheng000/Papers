
import mindspore
from mindspore import ops, Tensor
from mindnlp.transformers import T5ForConditionalGeneration, T5Tokenizer

import mindspore.nn as nn
tokenizer = T5Tokenizer.from_pretrained('t5-base')


class  MultiTaskT5(nn.Cell):
    def __init__(self, model: T5ForConditionalGeneration=None, rationale_weight=1.0, auto_weights=False):
        super().__init__()
        self.model = model
        self.rationale_weight = rationale_weight
        self.auto_weights = auto_weights
        if model is not None:
            self.config = self.model.config
        
        
    def construct(
        self,
        pred_input_ids, 
        pred_attention_mask,
        pred_labels,
        pred_labels_attention_mask,
        
        rationale_input_ids,
        rationale_attention_mask,
        rationale_labels,
        rationale_labels_attention_mask
    ):
    
        pred_output = self.model.construct(
            input_ids=pred_input_ids, 
            attention_mask=pred_attention_mask, 
            labels=pred_labels, 
            decoder_attention_mask=pred_labels_attention_mask
            )
        rationale_output = self.model.construct(
            input_ids=rationale_input_ids,
            attention_mask=rationale_attention_mask,
            labels=rationale_labels,
            decoder_attention_mask=rationale_labels_attention_mask
        )


        pred_loss = pred_output.loss
        rationale_loss = rationale_output.loss


        loss = pred_loss + self.rationale_weight * rationale_loss

        return loss
    
    def generate(
        self, 
        pred_input_ids, 
        rationale_input_ids,
        max_new_tokens = 256
    ):
        
        pred_outputs =  self.model.generate(input_ids=pred_input_ids, max_new_tokens=max_new_tokens)
        rationale_outputs = self.model.generate(input_ids=rationale_input_ids, max_new_tokens=max_new_tokens)

        return pred_outputs, rationale_outputs
    
    def from_pretrained(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    def save_pretrained(self,output_dir):
        self.model.save_pretrained(output_dir)






class  MultiTaskRLAlignT5(nn.Cell):
    def __init__(self, model: T5ForConditionalGeneration=None, align_score_model: T5ForConditionalGeneration=None, rationale_weight=1.0, align_weight=0.1, auto_weights=False, rl_sample=False):
        super().__init__()
        self.model = model
        self.align_score_model = align_score_model
        if self.align_score_model is not None:
            self.align_score_model.set_train(False)
        
        self.rationale_weight = rationale_weight
        self.align_weight = align_weight
        self.auto_weights = auto_weights
        
        self.rl_sample = rl_sample
        self.eos_token_id = 1
        
        
    def construct(
        self,
        pred_input_ids, 
        pred_attention_mask,
        pred_labels,
        pred_labels_attention_mask,
        
        rationale_input_ids,
        rationale_attention_mask,
        rationale_labels,
        rationale_labels_attention_mask
    ):
    
        pred_output = self.model.construct(
            input_ids=pred_input_ids, 
            attention_mask=pred_attention_mask, 
            labels=pred_labels, 
            decoder_attention_mask=pred_labels_attention_mask
            )
        rationale_output = self.model.construct(
            input_ids=rationale_input_ids,
            attention_mask=rationale_attention_mask,
            labels=rationale_labels,
            decoder_attention_mask=rationale_labels_attention_mask
        )
        
        
        pred_logits = pred_output.logits
        rationale_logits = rationale_output.logits 
        
        
        sum_log_generate_prob, rationale_pred_score, baseline_pred_score = self.generate_probs_and_reward(
            pred_logits, 
            rationale_logits, 
            
            pred_input_ids,
            pred_attention_mask, 
            pred_labels, 
            pred_labels_attention_mask, 
        )
        align_rl_loss = - (rationale_pred_score - baseline_pred_score) * sum_log_generate_prob
        align_rl_loss = align_rl_loss.mean()
        
        pred_loss = pred_output.loss
        rationale_loss = rationale_output.loss

        
        loss = pred_loss + self.rationale_weight * rationale_loss  + align_rl_loss * self.align_weight

        return loss
    
    
    
    def get_logits_format(self, output_logits): 
        batch_size, gen_length, _ = output_logits.shape
        gen_ids = ops.Argmax(output_type=mindspore.int64)(Tensor(output_logits))
        valid_length = []
        for i in range(batch_size): 
            _rat = gen_ids[i].tolist()
            if self.eos_token_id in _rat: 
                valid_length.append(_rat.index(self.eos_token_id))
            else: 
                valid_length.append(gen_length)
        _gen_attention_mask = [[1]*vl for vl in valid_length]

        gen_attention_mask = [gen + [0]*(gen_length-len(gen)) for gen in _gen_attention_mask] 
        gen_attention_mask = Tensor(gen_attention_mask, dtype=mindspore.int64) # [B, gen_length]

        return gen_ids, gen_attention_mask
    



    def generate_probs_and_reward(
        self, 
        pred_logits, rationale_logits, 
        
        pred_input_ids,
        pred_attention_mask,
        pred_labels, 
        pred_labels_attention_mask, 
    ):
        rationale_ids, gen_rationale_attention_mask = self.get_logits_format(rationale_logits)

        sample_rationale_ids = rationale_ids
        sample_gen_rationale_attention_mask = gen_rationale_attention_mask
        
        pred_ids, gen_pred_attention_mask = self.get_logits_format(pred_logits)  
         
        batch_size, seq_rationale_len, _ = rationale_logits.shape
        
        m = nn.Softmax(axis=2)
        probs = m(rationale_logits)

        sequence_probs = probs[mindspore.numpy.arange(batch_size).unsqueeze(1), mindspore.numpy.arange(seq_rationale_len).unsqueeze(0), rationale_ids.tolist()]
        log_sequence_probs = ops.Log()(sequence_probs)
        sum_generate_prob = ( log_sequence_probs * gen_rationale_attention_mask ).sum(-1) / gen_rationale_attention_mask.sum(-1)
        
        self.align_score_model.set_train(False)
        
        score_results = self.align_score_model(
            input_ids=sample_rationale_ids, 
            attention_mask=sample_gen_rationale_attention_mask,
            labels=pred_labels, 
            decoder_attention_mask=pred_labels_attention_mask 
        )
        
        score_pred_logits = score_results.logits 
        batch_size, gen_pred_len, _ = score_pred_logits.shape
        pred_probs = m(score_pred_logits)
        gen_pred_probs = pred_probs[mindspore.numpy.arange(batch_size).unsqueeze(1), mindspore.numpy.arange(gen_pred_len).unsqueeze(0), pred_ids.tolist()]
        pred_score = (ops.Log()(gen_pred_probs) * gen_pred_attention_mask).sum(-1)  / gen_pred_attention_mask.sum(-1)            #pred_probs = 

        # baseline score 
        baseline_score_results = self.align_score_model(
            input_ids=pred_input_ids, 
            attention_mask=pred_attention_mask, 
            labels=pred_labels,
            decoder_attention_mask=pred_labels_attention_mask
        )   
        baseline_score_pred_logits = baseline_score_results.logits 
        batch_size, baseline_pred_len, _ = baseline_score_pred_logits.shape
        baseline_pred_probs = m(baseline_score_pred_logits)
        baseline_gen_pred_probs = baseline_pred_probs[mindspore.numpy.arange(batch_size).unsqueeze(1), mindspore.numpy.arange(baseline_pred_len).unsqueeze(0), pred_ids.tolist()]
        baseline_pred_score = (ops.Log()(baseline_gen_pred_probs) * gen_pred_attention_mask).sum(-1)  / gen_pred_attention_mask.sum(-1)              #pred_probs = 
        
        return sum_generate_prob, pred_score, baseline_pred_score
    
    
    def generate(
        self, 
        pred_input_ids, 
        rationale_input_ids,
        max_new_tokens = 256
    ):
        
        pred_outputs =  self.model.generate(pred_input_ids, max_new_tokens=max_new_tokens)
        rationale_outputs = self.model.generate(rationale_input_ids, max_new_tokens=max_new_tokens)
        
        return pred_outputs, rationale_outputs
    
    
    def from_pretrained(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    def save_pretrained(self,output_dir):
        self.model.save_pretrained(output_dir)
