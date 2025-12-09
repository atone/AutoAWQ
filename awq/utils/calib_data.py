import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    samples = [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]
    return torch.cat(samples, dim=0)


def get_calib_audio_dataset(
    data: str,
    n_samples=128,
    max_seq_len=512,
    split="validation",
):
    dataset = load_dataset(data, split=split)
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.rename_column("text_ids", "input_ids").select_columns(
        ["input_ids", "speak_ids", "listen_ids"]
    )

    input_list = []
    speak_list = []
    listen_list = []
    max_input_len = 4096
    for data in dataset:
        input_list.append(torch.tensor(data["input_ids"])[:, :max_input_len])
        speak_list.append(torch.tensor(data["speak_ids"])[:, :max_input_len])
        listen_list.append(torch.tensor(data["listen_ids"])[:, :max_input_len])

    input_ids = torch.cat(input_list, dim=1)
    speak_ids = torch.cat(speak_list, dim=1)
    listen_ids = torch.cat(listen_list, dim=1)

    n_split = input_ids.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")

    input_list = [
        input_ids[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split) if i < n_samples
    ]
    speak_list = [
        speak_ids[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split) if i < n_samples
    ]
    listen_list = [
        listen_ids[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split) if i < n_samples
    ]

    return {
        "input_ids": torch.cat(input_list, dim=0),
        "speak_ids": torch.cat(speak_list, dim=0),
        "listen_ids": torch.cat(listen_list, dim=0),
    }
