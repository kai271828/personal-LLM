"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os
from typing import Union


class Prompter(object):
    __slots__ = ("template", "tokenizer", "_verbose")

    def __init__(
        self,
        template_name: str = "instruction",
        verbose: bool = False,
    ):
        # self.tokenizer = tokenizer
        self._verbose = verbose
        # self.cutoff_len = cutoff_len
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, encoding="utf-8") as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            prompt = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            prompt = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            prompt = f"{prompt}{label}"

        # prompt = f"{prompt}{self.tokenizer.eos_token}"
        if self._verbose:
            print(prompt)
        return prompt

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def get_user_prompt(self, output: str) -> str:
        return (
            output.split(self.template["response_split"])[0].strip()
            + "\n\n"
            + self.template["response_split"]
        )

    def generate_and_tokenize_prompt(self, data_sample: dict) -> dict:
        full_prompt = self.generate_prompt(
            data_sample["instruction"],
            data_sample["input"],
            data_sample["output"],
        )

        user_prompt = self.get_user_prompt(full_prompt)

        user_tokens_len = (
            len(
                self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.cutoff_len,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )
