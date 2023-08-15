def get_preprocessor(
    data_columns, prompter, tokenizer, cutoff_len, train_on_whole_sample: bool
):
    def generate_and_tokenize_data(data_sample):
        args = [data_sample[column] for column in data_columns]

        full_sample = prompter.generate_training_sample(*args)

        user_prompt = prompter.get_user_prompt(full_sample)

        user_tokens_len = len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=cutoff_len - 1,
                # padding="max_length",
            )["input_ids"]
        )

        full_tokens = (
            tokenizer(
                full_prompt,
                truncation=True,
                max_length=cutoff_len - 1,
                # padding="max_length",
            )["input_ids"]
            + tokenizer.eos_token_id
        )

        return {
            "input_ids": full_tokens,
            "labels": [-100] * user_tokens_len + full_tokens[user_tokens_len:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    return generate_and_tokenize_data
