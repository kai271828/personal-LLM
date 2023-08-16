"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os
from typing import Union, List


class Prompter(object):
    __slots__ = ("template", "data_columns", "_verbose")

    def __init__(
        self,
        template_name: str = "system_prompt",
        verbose: bool = False,
    ):
        self._verbose = verbose
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, encoding="utf-8") as fp:
            self.template = json.load(fp)
            self.data_columns = self.template["data_columns"]
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_training_sample(
        self,
        input: str,
        label: str,
    ) -> str:
        data_sample = self.template["prompt"].format(input=input) + label
        
        if self._verbose:
            print(data_sample)
        return data_sample

    def get_user_prompt(self, data_sample: str) -> str:
        if self._verbose:
            print(
                data_sample.split(self.template["response_split"])[0].strip()
                + self.template["response_split"]
            )
        return (
            data_sample.split(self.template["response_split"])[0].strip()
            + self.template["response_split"]
        )

    def get_response(self, data_sample: str) -> str:
        if self._verbose:
            print(data_sample.split(self.template["response_split"])[1].strip())
        return data_sample.split(self.template["response_split"])[1].strip()


class InstructionPrompter(Prompter):
    def __init__(
        self,
        template_name: str = "instruction",
        verbose: bool = False,
    ):
        super().__init__(template_name, verbose)

    def generate_training_sample(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: str = "",
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

        if self._verbose:
            print(prompt)
        return prompt
