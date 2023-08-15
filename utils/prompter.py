"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

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
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        input: str = None,
        label: Union[None, str] = None,
    ) -> str:
        """
        Returns the full prompt.
        If a label (response, or output) is provided, it's also appended.
        """
        prompt = self.template["prompt"].format(input=input)
        if label:
            prompt = f"{prompt}{label}"

        if self._verbose:
            print(prompt)
        return prompt

    def get_user_prompt(self, output: str) -> str:
        if self._verbose:
            print(
                output.split(self.template["response_split"])[0].strip()
                + self.template["response_split"]
            )
        return (
            output.split(self.template["response_split"])[0].strip()
            + self.template["response_split"]
        )

    def get_response(self, output: str) -> str:
        if self._verbose:
            print(output.split(self.template["response_split"])[1].strip())
        return output.split(self.template["response_split"])[1].strip()


class InstructionPrompter(Prompter):
    __slots__ = ("template", "_verbose")

    def __init__(
        self,
        template_name: str = "instruction",
        verbose: bool = False,
    ):
        super().__init__(template_name, verbose)

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

        if self._verbose:
            print(prompt)
        return prompt
