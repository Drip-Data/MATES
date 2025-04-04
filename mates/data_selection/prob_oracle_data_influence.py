from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          HfArgumentParser, set_seed, DataCollatorForSeq2Seq)
from datasets import Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import sys
sys.path.append('/root/autodl-tmp/MATES/')
from mates.train.get_training_dataset import get_training_dataset
from get_valid_dataset import get_dataset
from accelerate import Accelerator
from prob_args import ProbArguments
from tqdm import tqdm
import torch
import logging
import random

import os
import gc

logger = logging.getLogger(__name__)

def main():
    accelerator = Accelerator()
    parser = HfArgumentParser((ProbArguments,))
    prob_args, = parser.parse_args_into_dataclasses()
    set_seed(prob_args.seed)

    ckpt = prob_args.ckpt
    model_dir = prob_args.model_dir
    model_name = prob_args.model_name
    if ckpt != 0:
        model_path = os.path.join(model_dir, model_name, f"checkpoint-{ckpt}")
    else:
        model_path = os.path.join(model_dir, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_files = os.path.join(prob_args.data_dir, prob_args.train_files)
    # reference_files = os.path.join(prob_args.data_dir, prob_args.reference_files)   
    train_loader = get_oracle_dataloader(train_files, tokenizer)

    # 接入需要修改，因为路径会不一致，同时也需要处理自定义数据集
    valid_loader = get_reference_dataloader(prob_args.task, prob_args.ref_data_dir, tokenizer)

    train_loader, valid_loader = accelerator.prepare(train_loader, valid_loader)

    # 论文中 总数据量2.16亿，选了最后的10%，在这10%中选10000条作为oracle data(每张卡上有不同的种子shuffle)；reference data 1024条
    valid_loaders = [valid_loader]
    data = []

    for batch in tqdm(train_loader, desc=f"GPU {accelerator.process_index}", position=accelerator.process_index):
        
        # 加载上一个状态训练好的模型参数和优化器状态, 感觉会很耗时间，I/O开销会很大
        model = load_model(model_path, prob_args.torch_dtype)
        # resize embeddings if needed (e.g. for LlamaTokenizer)
        # 没有这一步
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, prob_args.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=prob_args.lr)

        # 无法加载优化器状态。 解决，但有待验证～
        optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))    # 学习率会覆盖吗？

        model, optimizer = accelerator.prepare(model, optimizer)

        scores = train(batch, model, optimizer, valid_loaders, accelerator)
        data.append(
            {
                "input_ids": batch.input_ids[0].cpu().numpy().tolist(),   # input_ids 和 bert 不兼容，看看论文怎么解决的，在下一阶段先解码。。。
                "scores": scores,
            }
        )

        del model, optimizer, batch
        gc.collect()
        torch.cuda.empty_cache()
    
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "scores": Sequence(Value("float32")),
        }
    )
    # out_influence/llama3-1B/bbh/checkpoint-xxxx/oracle_data_influence
    output_dir = os.path.join(prob_args.output_dir, prob_args.model_name, prob_args.task, f"checkpoint-{ckpt}", 'oracle_data_influence') 
    processed_ds = Dataset.from_list(data, features=features)
    processed_ds.save_to_disk(output_dir / str(accelerator.device), max_shard_size="1GB", num_proc=1)


def train(batch, model, optimizer, valid_loaders, accelerator):

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    with accelerator.no_sync(model):
        loss = model(input_ids=input_ids, labels=labels).loss
        accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    return evaluate(model, valid_loaders, accelerator)

@torch.no_grad()
def evaluate(model, valid_loaders, accelerator):
    model.eval()
    losses = []
    for val_dataloader in valid_loaders:
        total_loss = 0.0
        cnt = 0
        for i, batch in enumerate(val_dataloader):
            # logits = model(input_ids)
            # loss += chunked_cross_entropy(
            #     logits[:, :-1, :],
            #     labels[:, 1:],
            #     chunk_size=0,
            # )
            output = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = output.loss
            total_loss += loss.item()
            cnt += 1
            
            del loss, output, batch
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = total_loss / cnt
        losses.append(avg_loss)
    model.train()
    return losses


def get_oracle_dataloader(train_files, tokenizer, max_seq_length=2048, percentage=0.01, seed=42):

    # 按照论文的逻辑进行了修改，选取后10%的数据作为oracle data
    held_out_data = get_training_dataset(train_files, tokenizer,max_seq_length, percentage, seed=seed)


    # print(held_out_data[0]) 还有一些其他信息需要去除
    if "dataset" in held_out_data.features:
        held_out_data = held_out_data.remove_columns(
            ["dataset", "id", "messages"])
            

    for index in random.sample(range(len(held_out_data)), 1):
        logger.info(
            f"Sample {index} of the training set: {held_out_data[index]}.")
    dataloader = DataLoader(held_out_data,
                            batch_size=1,  # When getting gradients, we only do this single batch process
                            collate_fn=DataCollatorForSeq2Seq(
                                tokenizer=tokenizer, padding="longest"))

    return dataloader

def get_reference_dataloader(task, reference_files, tokenizer, max_seq_length=2048):

    ######### 是否需要加入task 字段用于后续文件夹的保存？ 加了##########
    reference_data = get_dataset(
        task,
        data_dir=reference_files,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        use_chat_format=True,
        chat_format="tulu",
        shuffle=False,
        seed=43,
    )
    # print(reference_data[0])
    
    for index in random.sample(range(len(reference_data)), 1):
        logger.info(
            f"Sample {index} of the reference set: {reference_data[index]}.")
        
    dataloader = DataLoader(reference_data,
                            batch_size=2,  # When getting gradients, we only do this single batch process
                            collate_fn=DataCollatorForSeq2Seq(
                                tokenizer=tokenizer, padding="longest"))
    
    return dataloader

def load_model(model_name_or_path, torch_dtype):

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype)
        
    return model

def get_optimizer_grouped_parameters(model, weight_decay):

    no_decay = ["bias", "LayerNorm.weight"]     # 不确定,lora没有问题，但如果是全量微调则需要检查别的，参考trainer中的设置
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,  # 需与训练时的权重衰减值一致     
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

if __name__ == "__main__":
    main()