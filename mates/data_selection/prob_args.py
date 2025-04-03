from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ProbArguments:

    # 没有default的写在前面，有default的写在后面，否则报错！！！！！！！！！

    model_dir: str = field(
        metadata={"help": "The checkpoint directory."}
    )
    output_dir: str = field(
        metadata={"help": "The output influence score directory."}
    )

    train_files: str = field(
        metadata={"help": "The directory of the held out data."}
    )

    task: str = field(
        metadata={"help": "The task name."}
    )

    reference_files: str = field(
        metadata={"help": "The directory of the reference data."}
    )

    ckpt: int = field(
        default=1000, metadata={"help": "The checkpoint to load from. "}
    )

    
    seed: int = field(
        default=11, metadata={"help": "A seed for reproducible training."}
    )
    torch_dtype: Optional[str] = field(
        default=torch.bfloat16,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
       },
    )

    model_name: str = field(
        default="llama3-1B",
        metadata={"help": "The model name."}
    )

    

