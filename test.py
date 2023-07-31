from utils.prompter import Prompter
from datasets import load_dataset

data = load_dataset(
    "json",
    data_files="C:\\Users\\yikailiao\\Desktop\\Formosan-Sika-Deer\\data\\train_zhtw.json",
)

print(data)

# prompter = Prompter()


# full_prompt = prompter.generate_prompt(
#     "test1",
#     "test2",
#     "test3",
# )

# user_prompt = prompter.get_user_prompt(full_prompt)
# print(user_prompt)
