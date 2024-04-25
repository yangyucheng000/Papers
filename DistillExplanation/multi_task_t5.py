import logging
import os
import sys 
import json
from datetime import datetime
import random

from mindspore import nn
from mindnlp.engine import Trainer
from mindnlp.engine.callbacks import CheckpointCallback
from mindnlp.transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils import set_all_seed, LinearWithWarmUp
from src.load_datasets import  load_t5_dataset
from src.t5_model import  MultiTaskT5, MultiTaskRLAlignT5

from config.configs import DistillConfig


random.seed(10)

logger = logging.getLogger(__name__)



PREFIX_CHECKPOINT_DIR = "checkpoint"






def main():

    args = DistillConfig.load("align_configs.json")

    model_args = args.model_config
    data_args = args.data_config
    training_args = args.training_config

    print(model_args)
    print(data_args)
    print(training_args)

    set_all_seed(training_args.seed)

    # Create output directory if do training.
    if training_args.do_train: 
        # create a save directory and a logfile
        save_path = training_args.output_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        training_args.output_dir = os.path.join(
            save_path, 
            f'{data_args.task_name}/T5-base/score_model_{model_args.score_model}/MultiTask_{model_args.multi_task}/RL_align_{model_args.align_with_rl}', 
            datetime.now().strftime("%m%d%y_%H%M%S")
        )
        training_args.logging_dir = training_args.output_dir
        
        assert os.path.exists(save_path)
        assert not os.path.exists(training_args.output_dir)
        os.makedirs(training_args.output_dir)
        
        handlers = [
            logging.FileHandler(os.path.join(training_args.output_dir, "logger.log")),
            logging.StreamHandler(),
        ]
        
    else:
        # don't overwrite existing logfile or create new directory
        training_args.output_dir = model_args.trained_ckpt_dir
        handlers = [logging.StreamHandler()]
    
    
    # 
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s",
        training_args.device,
        training_args.n_gpu,
    )
    logger.info("Save path: %s" % training_args.output_dir)
    
    

    
    
    
    # set gradient accumulation steps to always use batch size == 64
    training_args.gradient_accumulation_steps = int(
        64 / training_args.per_device_train_batch_size
    )
    
    
    # write command and args to file if training
    if training_args.do_train:
        # write command and args to file
        with open( os.path.join(training_args.output_dir, "commandline_args.txt"), "w" ) as f:
            f.write("Command:\n")
            f.write("\n".join(sys.argv[1:]))
            f.write("Training args:\n")
            # make training_args dict writeable
            tmp = training_args.__dict__
            tmp.pop("__cached__setup_devices", None)
            tmp.pop("evaluation_strategy", None)
            tmp.pop("lr_scheduler_type", None)
            tmp.pop("logging_strategy", None)
            tmp.pop("save_strategy", None)
            json.dump(tmp, f, indent=2)
            f.write("Data args:\n")
            json.dump(data_args.__dict__, f, indent=2)
            f.write("Model args:\n")
            json.dump(model_args.__dict__, f, indent=2)
    
    

    # Load pretrained model and tokenizer
    logger.info("Loading pretrained tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    if data_args.generations_filepath is None:  
        
        if model_args.trained_ckpt_dir:
            # if existing trained model checkpoint
            """ """

        else:
            # no existing trained model checkpoint, so need training
            logger.info("Loading pretrained model")
            logger.info(model_args.pretrained_model)

            base_model = T5ForConditionalGeneration.from_pretrained(model_args.pretrained_model)
    
    else:
        base_model = None
    
    
    
    if model_args.align_with_rl: 

        align_score_model = T5ForConditionalGeneration.from_pretrained(model_args.score_model_path)
        model = MultiTaskRLAlignT5(
            base_model, 
            align_score_model, 
            rationale_weight=model_args.rationale_weight,
            auto_weights=model_args.auto_weights, 
            align_weight=model_args.align_weight, 
            rl_sample=model_args.rl_sample
        )
    
    elif model_args.multi_task: 
        model = MultiTaskT5(base_model, rationale_weight=model_args.rationale_weight, auto_weights=model_args.auto_weights)

    else: 
        model = T5ForConditionalGeneration.from_pretrained(model_args.pretrained_model)
        
        
    # Get datasets
    if data_args.task_name == 'winogrande': 
        data_dir = 'data/winogrande/data'
    elif data_args.task_name == 'copa':
        data_dir = 'data/copa/data'
    else:
        raise ValueError(f"task_name {data_args.task_name} not supported")    
    
    train_dataset, eval_dataset, test_dataset = load_t5_dataset(
        data_dir=data_dir, 
        multi_task=model_args.multi_task, 
        score_model=model_args.score_model,
    )    

        
    logger.info("****LOG****")
    logger.info(len(train_dataset))
    logger.info(len(eval_dataset))
    logger.info(len(test_dataset))
    

    #### TRAIN ####
    
    do_train = training_args.do_train
    if do_train:
        num_epochs = training_args.num_epoch
        warmup_steps = 2000
        learning_rate = training_args.learning_rate
        num_training_steps = num_epochs * train_dataset.get_dataset_size()

        lr_scheduler = LinearWithWarmUp(learning_rate=learning_rate, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_scheduler)
        ckpoint_cb = CheckpointCallback(save_path=f'checkpoint/{data_args.task_name}/T5-base/score_model_{model_args.score_model}/MultiTask_{model_args.multi_task}/RL_align_{model_args.align_with_rl}', ckpt_name='expl', epochs=1, keep_checkpoint_max=3)

        
        trainer = Trainer(network=model, train_dataset=train_dataset,
                    epochs=num_epochs, optimizer=optimizer, callbacks=[ckpoint_cb])

        
        trainer.run()  
        
        model.save_pretrained(os.path.join(training_args.output_dir, 'saved_model'))

    
    
    
    if not model_args.score_model:
        
        saved_model_path = os.path.join(training_args.output_dir, 'saved_model')
        print(f"model saved to {saved_model_path}")
        #### TEST ####

        if model_args.multi_task:
            if model_args.align_with_rl: 
                model = MultiTaskRLAlignT5()
                model.from_pretrained(model_path=saved_model_path)
            else:
                model = MultiTaskT5()
                model.from_pretrained(model_path=saved_model_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(saved_model_path)

        model.set_train(False)

        compute_multi_task_metrics(model, tokenizer, test_dataset, save_dir=os.path.dirname(saved_model_path), model_args=model_args, split='test')
        
        


def compute_multi_task_metrics(model, tokenizer, test_dataset, save_dir, model_args, split='test'):
    """"""
    fname = os.path.join(save_dir, "%s_generations.txt" % split)
    analysis_file = os.path.join(save_dir, "%s_posthoc_analysi.txt" % split)
    if os.path.isfile(fname):
        fname = fname.split(".txt")[0] + "_1.txt"
    if os.path.isfile(analysis_file):
        analysis_file = analysis_file.split(".txt")[0] + "_1.txt"

    
    pred_texts = []
    label_texts = []

    with open(fname, "w") as fgen:
        
        for batch in test_dataset.create_dict_iterator():

            pred_outputs, rationale_outputs = model.generate(
                pred_input_ids = batch['pred_input_ids'], 
                rationale_input_ids = batch['rationale_input_ids'],
                max_new_tokens = 256
            )
            
            decoded_pred_outputs = tokenizer.batch_decode(pred_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_rationale_outputs = tokenizer.batch_decode(rationale_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            decoded_pred_inputs = tokenizer.batch_decode(batch['pred_input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_rationale_inputs = tokenizer.batch_decode(batch['rationale_input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            fix_batch_pred_labels = (batch['pred_labels'] != -100) * batch['pred_labels']
            fix_batch_rationale_labels = (batch['rationale_labels'] != -100) * batch['rationale_labels']
            decoded_pred_labels = tokenizer.batch_decode(fix_batch_pred_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_rationale_labels = tokenizer.batch_decode(fix_batch_rationale_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            batch_size, _ = pred_outputs.shape

            for i in range(batch_size):

                fgen.write(f"Pred Input: {decoded_pred_inputs[i]}\n")
                fgen.write(f"Pred Label: {decoded_pred_labels[i]}\n")
                fgen.write(f"Pred Output: {decoded_pred_outputs[i]}\n")
                
                fgen.write(f"Rationale Input: {decoded_rationale_inputs[i]}\n")
                fgen.write(f"Rationale Label: {decoded_rationale_labels[i]}\n")
                fgen.write(f"Rationale Output: {decoded_rationale_outputs[i]}\n")
                
                fgen.write(f"The prediction is correct: {decoded_pred_labels[i] == decoded_pred_outputs[i]}\n")
                
                fgen.write(f"="*20 + "\n\n")
                fgen.write("\n\n")
                
                pred_texts.append(decoded_pred_outputs[i])
                label_texts.append(decoded_pred_labels[i])
            
            
    fgen.close()
    acc = sum([1 for i in range(len(pred_texts)) if pred_texts[i] == label_texts[i]]) / len(pred_texts)
    print(acc)    

    result = {'accuracy':  acc}
    with open(os.path.join(save_dir,'results.txt'), 'w') as f: 
        json.dump(result, f, indent=4)     

    
if __name__ == "__main__":
    main()
