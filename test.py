from utils.prompter import Prompter

prompter = Prompter()

full_prompt = prompter.generate_prompt(
    "test1",
    "test2",
    "test3",
)

user_prompt = prompter.get_user_prompt(full_prompt)
print(user_prompt)
