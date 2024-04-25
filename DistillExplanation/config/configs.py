import json
import os
from typing import Optional, Tuple

from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from mindformers.trainer import Trainer, TrainingArguments



@dataclass
class TrainingConfig(TrainingArguments):
    "Configuration for training"
    do_train: bool=True
    output_dir: str='./'
    seed: int=0
    
    
    learning_rate: float=2e-5
    batch_size: int=16
    num_epoch: int=5
    use_cuda: bool=True
    optimizer_type: str='adam'
    
    logging_dir: Optional[str]=None
    
    gradient_accumulation_steps: Optional[int]=None   
    
    device: str='cuda'
    n_gpu: int=1
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8


@dataclass
class ModelConfig(DataClassJsonMixin):
    "Configuration for training"
    pretrained_model: str
    
    trained_ckpt_dir: Optional[str]=None
    
    multi_task: bool = field(
        default=False,
        metadata={"help": "For multi-task I->O, I->R model."},
    )
    score_model: bool = field(
        default=False,
        metadata={"help": "For score model."},
    )
    score_model_path: str=None
    
    rationale_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "rationale loss weight"
        },
    )
    auto_weights: bool = field(
        default=False,
        metadata={"help": "For multi-task, use auto weigth."},
    )
    align_with_rl: bool = field(
        default=False,
        metadata={"help": "Use REINFORCE for rationale label align."},
    )
    align_weight: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "RL align weight"
        },
    )
    
    rl_sample: bool = field(
        default=False,
        metadata={"help": "Use REINFORCE for rationale label align."},
    )
    
    rationale_only: bool = field(
        default=False,
        metadata={
            "help": "Only produce rationales and not labels (first model in pipeline)"
        },
    )
    
    label_only: bool = field(
        default=False,
        metadata={"help": "Only produce labels and not rationales (I-->O baseline)"},
    )
    
    explanation_first: bool = field(
        default=False,
        metadata={"help": "explanation answer: xxx"},
    )
    
    
    

    
    
    
    


@dataclass
class DataTraingConfig(DataClassJsonMixin):
    task_name: str
    train_predict: bool
    test_predict: bool 
    
    label_on_ground_truth: bool = field(
        default=False, 
        metadata={"help": "For multi-task, Use ground truth label as the prediction task label."}
    )
    rationale_on_ground_truth: bool = field(
        default=False, 
        metadata={"help": "For multi-task, when label_in_rationale_input is true, Use ground truth label as the rationale input ."}
    )
    label_in_rationale_input: bool = field(
        default=False, metadata={"help": "For multi-task, add prediction in the rationale input, to guide model generate rationale for the prediction."}
    )
    
    generate_label_in_rationale_input: bool = field(
        default=False, metadata={"help": "For multi-task, when label_in_rationale_input is true, Use model generated prediction answer as the rationale input "}  # TODO
    )
    
    two_step_data: bool = field(
        default=False,
        metadata={"help": "For two-step models I->O, IO->R ."},
    )
    cot_data: bool = field(
        default=False,
        metadata={"help": "For two-step models with CoT: I->O, I->RO ."},
    )
    
    on_ground_truth_label: bool = field(
        default=False, metadata={"help": "For I->OR, Use ground truth label"}
    )
    

    
    generations_filepath: Optional[str]=None

    las_data: bool = field(
        default=False, metadata={"help": "For training las model. "}  # 
    )



@dataclass
class DistillConfig(DataClassJsonMixin):
    training_config: TrainingConfig
    model_config: ModelConfig             # Model config including hyperparams
    data_config: DataTraingConfig

    

    @classmethod
    def _load_from_file(cls, config: str) -> "DistillConfig":
        with open(config, "r") as f:
            config_json = json.load(f)
        return cls.from_dict(config_json)

    @classmethod
    def load(cls, config: str) -> "DistillConfig":
        if os.path.exists(config):
            configs = cls._load_from_file(config)
            return configs

