import logging
import os
import random
import sys
import datasets
import transformers
from get_training_dataset import get_training_dataset
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from arguments import (TrainingArguments, DataArguments, ModelArguments,
                       get_data_statistics)
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          HfArgumentParser, set_seed,
                          Trainer, DataCollatorForSeq2Seq)


logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # get training dataset
    # 根据训练的步数决定训练的dataset
    # ckpt = 0, 就是随机选取一个batch
    if training_args.ckpt == 0:
        print("ckpt is 0, using all training dataset, randomly select a batch every step.")
        train_files = data_args.train_files

    # 选择ckpt对应的的数据集，在数据筛选的过程中会保存一个json文件
    else:
        print("ckpt is not 0, using the data selected from last ckpt.")
        train_files = data_args.data_dir + f"{training_args.ckpt}"

    train_dataset = get_training_dataset(train_files, 
                                   tokenizer, 
                                   data_args.max_seq_length, 
                                   data_args.percentage, 
                                   data_args.sample_data_seed)


    # 待完善，选择与上面数据相对应的权重加载训练
    # when ckpt==0，chose the pretrained model
    if training_args.ckpt == 0:
        print("Loading pretrained model...")
        model_path = model_args.model_name_or_path
        resume_checkpoint = None

    # ckpt != 0时，chose the ckpt model
    else:
        checkpoint_path = os.path.join(training_args.output_dir, f"checkpoint-{training_args.ckpt}")
        print(f"Loading model from checkpoint: {checkpoint_path}")
        resume_checkpoint = checkpoint_path
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_args.torch_dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])
            

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")


    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)


    # training 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"),
        output_dir=training_args.output_dir,
        max_steps=training_args.ckpt + training_args.update_steps,
        # report_to="tensorboard"
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()