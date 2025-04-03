from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          HfArgumentParser, set_seed, DataCollatorForSeq2Seq)
from datasets import Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import sys
sys.path.append('/root/autodl-tmp/MATES/')
from mates.train.get_training_dataset import get_training_dataset
from accelerate import Accelerator
from prob_args import ProbArguments
import torch
import logging
import random
import os

logger = logging.getLogger(__name__)

def main():

    parser = HfArgumentParser((ProbArguments,))
    prob_args, = parser.parse_args_into_dataclasses()
    set_seed(prob_args.seed)

    ckpt = prob_args.ckpt
    model_dir = prob_args.model_dir
    if ckpt != 0:
        model_path = os.path.join(model_dir, f"checkpoint-{ckpt}")
    else:
        model_path = model_dir

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_loader = get_oracle_dataloader(prob_args.train_files, tokenizer)

    valid_loader = get_reference_dataloader(prob_args.reference_files, tokenizer)
    # print(len(train_loader))
    # for batch in train_loader:
    #     print(batch)
    #     break

    valid_loaders = [valid_loader]
    data = []
    for batch in train_loader:

        # 加载上一个状态训练好的模型参数和优化器状态, 感觉会很耗时间
        model = load_model(model_path, prob_args.torch_dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))    # 学习率会覆盖吗？

        scores = train(batch, model, optimizer, valid_loaders)
        data.append(
            {
                "input_ids": batch.input_ids[0].cpu().numpy().tolist(),   # input_ids 和 bert 不兼容，看看论文怎么解决的
                "scores": scores,
            }
        )

        del model, optimizer, batch
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
    processed_ds.save_to_disk(output_dir / str(rank), max_shard_size="1GB", num_proc=1)


def train(batch, model, optimizer, valid_loaders):

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return evaluate(model, valid_loaders)


def evaluate(model, valid_loaders):
    model.eval()
    losses = []
    for val_dataloader in valid_loaders:
        loss = torch.tensor(0.0, device=device)
        cnt = 0
        for batch in val_dataloader:
            # logits = model(input_ids)
            # loss += chunked_cross_entropy(
            #     logits[:, :-1, :],
            #     labels[:, 1:],
            #     chunk_size=0,
            # )
            loss += model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
            cnt += 1
        loss = loss / cnt
        losses.append(loss.item())
    model.train()
    return losses


def get_oracle_dataloader(train_files, tokenizer, max_seq_length=2048, percentage=0.1, seed=43):

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

def get_reference_dataloader(reference_files, tokenizer, max_seq_length=2048):

    ######### 是否需要加入task 字段用于后续文件夹的保存？ ##########
    reference_data = get_training_dataset(reference_files, tokenizer,max_seq_length)

def load_model(model_name_or_path, torch_dtype):

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype)
        
    return model



if __name__ == "__main__":
    main()