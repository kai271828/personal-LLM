def get_preprocessor(
    prompter, tokenizer, cutoff_len, train_on_whole_sample: bool = False
):
    def generate_and_tokenize_data(data_sample):
        args = [data_sample[column] for column in prompter.data_columns]

        full_sample = prompter.generate_training_sample(*args)

        full_tokens = tokenizer(
            full_sample, truncation=True, max_length=cutoff_len - 1, padding=False
        )
        full_tokens["input_ids"].append(tokenizer.eos_token_id)
        full_tokens["attention_mask"].append(1)
        full_tokens["labels"] = full_tokens["input_ids"].copy()

        if not train_on_whole_sample:
            user_prompt = prompter.get_user_prompt(full_sample)
            len_user_tokens = len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=cutoff_len - 1,
                    padding=False,
                )["input_ids"]
            )

            full_tokens["labels"] = [-100] * len_user_tokens + full_tokens["labels"][
                len_user_tokens:
            ]

        return full_tokens

    return generate_and_tokenize_data
