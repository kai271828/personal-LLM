# TODO: add required libraries
import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from utils.prompter import Prompter


def main(
    # model/data parameters
    base_model: str = "bigscience/bloomz-560m",  # default model
    data_path: str = "",  # required argument
    output_dir: str = "./outputs",
    cache_dir: str = "./cache",
    quantization: Union[str, None] = None,
    nested_quant: bool = False,
    bnb_4bit_quant_type: str = "fp4",
    bnb_4bit_compute_dtype: str = "float32",
    deepspeed: Union[str, None] = None,
    # peft parameters
    tuner: Union[str, None] = None,
    # lora hyperparameters
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Union[List[str], None] = None,
    # ia3 hyperparameters
    ia3_target_modules: Union[List[str], None] = None,
    ia3_feedforward_modules: Union[List[str], None] = None,
    # prompt tuning hyperparameters
    # prefix tuning hyperparameters
    # p-tuning hyperparameters
    # llm hyperparameters
    add_eos_token: bool = True,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    use_fp16: bool = False,
    use_bf16: bool = False,
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit: int = 3,
    gradient_checkpointing: bool = False,
    group_by_length: bool = False,
    optim: str = "adamw_torch",
    # wandb params
    # use_wandb: bool = False,
    # wandb_project: str = "",
    # wandb_run_name: str = "",
    # wandb_watch: str = "",  # options: false | gradients | all
    # wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: Union[
        str, None
    ] = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "instruction",  # The prompt template to use, will default to instruction.
    local_rank: int = 0,
):
    # Check hyperparameters
    # TODO: add the parameters and their checking if needed
    assert (
        data_path
    ), "Please specify a --data_path, e.g. --data_path='./data/train.json'"
    assert quantization in [
        "4bit",
        "8bit",
        None,
    ], "--quantization only supports '4bit', '8bit', or None."
    assert bnb_4bit_quant_type in [
        "fp4",
        "nf4",
    ], "--bnb_4bit_quant_type only supports 'fp4' or 'np4'."
    assert bnb_4bit_compute_dtype in [
        "float32",
        "float16",
        "bfloat16",
    ], "--bnb_4bit_compute_dtype only supports 'float32', 'float16', or 'bfloat16'"
    assert tuner in [
        "LoRA",
        "IA3",
        "Prompt",
        "Prefix",
        "P-tuning",
        None,
    ], "--tuner only supprts 'LoRA', 'IA3', 'Prompt', 'Prefix', or 'P-tuning', or None."
    assert prompt_template_name in [
        None,
        "instruction",
    ], "--prompt_template_name only supports 'instruction' or None"
    assert optim in [
        "adamw_torch",
        "adamw_hf",
        "adafactor",
    ], "--optim only support 'adamw_torch', 'adamw_hf', or 'adafactor"

    if quantization:
        assert tuner, "Training quantized weights directly is not supported."

    if lora_target_modules is None and tuner == "LoRA":
        if "bloom" in base_model:
            lora_target_modules = ["query_key_value"]
        elif "mt" in base_model:
            lora_target_modules = ["q", "v"]
        elif "llama" in base_model:
            lora_target_modules = ["q_proj", "v_proj"]
        elif "Falcon" in base_model or "falcon" in base_model:
            lora_target_modules = ["query_key_value"]

    if ia3_target_modules is None and tuner == "IA3":
        if "bloom" in base_model:
            ia3_target_modules = ["query_key_value", "mlp.dense_4h_to_h"]
        elif "mt" in base_model:
            ia3_target_modules = ["k", "v", "wi_1"]
        elif "llama" in base_model:
            ia3_target_modules = ["k_proj", "v_proj", "down_proj"]
        elif "Falcon" in base_model or "falcon" in base_model:
            ia3_target_modules = ["query_key_value"]

    if ia3_feedforward_modules is None and tuner == "IA3":
        if "bloom" in base_model:
            ia3_feedforward_modules = ["mlp.dense_4h_to_h"]
        elif "mt" in base_model:
            ia3_feedforward_modules = []
        elif "llama" in base_model:
            ia3_feedforward_modules = ["down_proj"]
        elif "Falcon" in base_model or "falcon" in base_model:
            ia3_feedforward_modules = ["query_key_value"]

    # Print trainging information
    # TODO: show the parameters set

    # multi-node setting
    device_map = "auto"
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # # Only overwrite environ if wandb param passed
    # if use_wandb:
    #     if len(wandb_project) > 0:
    #         os.environ["WANDB_PROJECT"] = wandb_project
    #     if len(wandb_watch) > 0:
    #         os.environ["WANDB_WATCH"] = wandb_watch
    #     if len(wandb_log_model) > 0:
    #         os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Load the data
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Prepare tokenizer
    if "llama" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # Allow batched inference

    # Process the data
    # TODO: examine the processing and complete it
    prompter = Prompter(prompt_template_name)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )

        user_prompt = prompter.get_user_prompt(full_prompt)

        user_tokens_len = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=cutoff_len + 1,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )

        full_tokens = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len + 1,
            padding="max_length",
        )["input_ids"][:-1]

        return {
            "input_ids": full_tokens,
            "labels": [-100] * user_tokens_len + full_tokens[user_tokens_len:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=9527
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Prepare model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True if quantization == "4bit" else False,
        load_in_8bit=True if quantization == "8bit" else False,
        bnb_4bit_use_double_quant=True
        if quantization == "4bit" and nested_quant
        else False,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

    if "llama" in base_model:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            # torch_dtype="auto",  # may be bug
            cache_dir=cache_dir,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            # torch_dtype="auto",  # may be bug
            cache_dir=cache_dir,
            device_map=device_map,
            trust_remote_code=True,
        )

    if not (use_fp16 or use_bf16 or quantization):
        model.half()
        print("cast model to half.")

    if quantization and tuner:
        model = prepare_model_for_kbit_training(model)

    # TODO: implement the tuner
    if tuner == "LoRA":
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif tuner == "IA3":
        peft_config = IA3Config(
            peft_type="IA3",
            target_modules=ia3_target_modules,
            feedforward_modules=ia3_feedforward_modules,
            task_type="CAUSAL_LM",
        )
    elif tuner == "Prompt":
        raise NotImplementedError
    elif tuner == "Prefix":
        raise NotImplementedError
    elif tuner == "P-tuning":
        raise NotImplementedError

    if tuner:
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    # TODO: review resume function
    if resume_from_checkpoint:
        raise NotImplementedError
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if tuner:
        model.print_trainable_parameters()

    # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Trainer setting
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        optim=optim,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,  # "wandb" if use_wandb else None,
        run_name=None,  # wandb_run_name if use_wandb else None,
        deepspeed=deepspeed,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    # Train
    if tuner:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        print("perform torch compile.")

    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(main)
