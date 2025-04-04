from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments
import torch



@dataclass
class TrainingArguments(TrainingArguments):

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    seed: Optional[int] = field(
        default=11, metadata={"help": "A seed for reproducible training."}
    )
    # 根据ckpt选取数据
    ckpt: int = field(
        default=0, metadata={"help": "The checkpoint to load from. "}
    )

    update_steps: int = field(
        default=1000, metadata={"help": "The update steps of data influence model."}
    )





@dataclass
class DataArguments:

    data_dir: str = field(
        metadata={"help": "The input data dir."}
    )
    train_files: str = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    
    max_seq_length: int = field(
        default=2048, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                                        "than this will be truncated, sequences shorter will be padded."}
    )

    sample_data_seed: int = field(
        default=42, metadata={"help": "The seed to use when sampling data for training."}
    )

    percentage: float = field(
        default=1.0, metadata={"help": "The percentage of data to use for training."}
    )



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_name: str = field(
        default="llama3-1B",
        metadata={"help": "The model name."}
    )

    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA for fine-tuning."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
   )
    lora_r: int = field(
        default=8, metadata={"help": "The r of LoRA."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "The alpha of LoRA."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout of LoRA."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"], metadata={"help": "The target modules of LoRA."}
    )
    
def get_data_statistics(lm_datasets):
    """ Get the data statistics of the dataset. """
    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]

        completion_lens = []
        for labels in examples["labels"]:
            com_len = (torch.tensor(labels) > -1).sum()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(
            f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}")
