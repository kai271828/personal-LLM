import os
import sys
from typing import Union

import fire
import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

from utils.prompter import Prompter


def main(
    base_model: str = "",
    adapter: Union[str, None] = None,
    quantization: Union[str, None] = None,
    nested_quant: bool = False,
    bnb_4bit_quant_type: str = "fp4",
    bnb_4bit_compute_dtype: str = "float32",
    prompt_template_name: str = "instruction",
    cache_dir: str = "./cache",
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 10,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    stream_output=False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='bigscience/bloomz-560m'"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    prompter = Prompter(prompt_template_name)

    if "llama" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)

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
            torch_dtype="auto",
            cache_dir=cache_dir,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype="auto",
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=True,
        )

    if adapter:
        model = PeftModel.from_pretrained(
            model,
            adapter,
            torch_dtype="auto",
        )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    while True:
        instruction_ = input("Please enter instruction: ")
        input_ = input("Please enter the corresponding input (or just skip): ")

        prompt = prompter.generate_prompt(instruction_, input_)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            s = generation_output.sequences[0]
            print(s)
            output = tokenizer.decode(s)
            print(output)


if __name__ == "__main__":
    fire.Fire(main)
