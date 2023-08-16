from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

from utils.prompter import Prompter, InstructionPrompter
from utils.preprocessing import get_mapping

data = load_dataset(
    "json",
    data_files="C:\\Users\\yikailiao\\Desktop\\Formosan-Sika-Deer\\data\\data_0814.json",
)

prompter = InstructionPrompter(verbose=True)
tokenizer = LlamaTokenizer.from_pretrained("ziqingyang/chinese-alpaca-2-7b")
generate_and_tokenize_prompt = get_mapping(
    ["instruction", "input", "output"], prompter, tokenizer, 512
)

dataset = data["train"].map(generate_and_tokenize_prompt)

print(dataset[0])

# full_prompt = prompter.generate_prompt(
#     data_sample["instruction"],
#     data_sample["output"],
# )

# # full_prompt = prompter.generate_prompt(
# #     data_sample["instruction"],
# #     data_sample["input"],
# #     data_sample["output"],
# # )
# print(
#     "======================================================================================"
# )
# user_prompt = prompter.get_user_prompt(full_prompt)
# print(
#     "======================================================================================"
# )
# response = prompter.get_response(full_prompt)
