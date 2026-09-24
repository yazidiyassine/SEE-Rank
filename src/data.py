"""
Dataset loading, deduplication, and preprocessing utilities.
"""

from typing import Tuple, Dict, Any
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


def load_data(
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """
    Load and preprocess the instruction fine-tuning dataset with strict split isolation.

    Args:
        tokenizer: PreTrainedTokenizer for sequence encoding.
        config: Configuration dictionary specifying dataset size, sequence length, and loss scope.

    Returns:
        Tuple containing:
            - train_ds: Formatted PyTorch Dataset for training.
            - val_ds: Formatted PyTorch Dataset for validation perplexity.
            - manifest: Metadata dictionary documenting sample counts, deduplication, and split overlap.
    """
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    df = ds.to_pandas()

    n_before = len(df)
    df = df.drop_duplicates(subset=["instruction", "output"]).reset_index(drop=True)
    n_after = len(df)

    df = df.sample(frac=1.0, random_state=config["DATA_SEED"]).reset_index(drop=True)
    needed = config["TRAIN_SAMPLES"] + config["VAL_SAMPLES"]
    df = df.iloc[:min(needed, len(df))].reset_index(drop=True)

    train_df = df.iloc[:config["TRAIN_SAMPLES"]].reset_index(drop=True)
    val_df = df.iloc[config["TRAIN_SAMPLES"]:min(needed, len(df))].reset_index(drop=True)

    overlap = set(train_df["instruction"]).intersection(set(val_df["instruction"]))

    prompt_template = "### Instruction:\n{inst}\n\n### Response:\n{out}"

    def tokenize_batch(batch):
        insts, outs = batch["instruction"], batch["output"]
        texts = [prompt_template.format(inst=i, out=o) for i, o in zip(insts, outs)]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=config["MAX_SEQ_LEN"],
            padding="max_length",
            return_tensors="pt"
        )
        labels = enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        if config.get("LOSS_SCOPE") == "response_only":
            prefix_texts = [
                "### Instruction:\n{inst}\n\n### Response:\n".format(inst=i)
                for i in insts
            ]
            prefix_lens = [
                len(tokenizer(p, truncation=True, max_length=config["MAX_SEQ_LEN"])["input_ids"])
                for p in prefix_texts
            ]
            for row_idx, plen in enumerate(prefix_lens):
                labels[row_idx, :min(plen, labels.shape[1])] = -100

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }

    train_ds = Dataset.from_pandas(train_df).map(
        tokenize_batch,
        batched=True,
        batch_size=500,
        remove_columns=train_df.columns.tolist()
    )
    val_ds = Dataset.from_pandas(val_df).map(
        tokenize_batch,
        batched=True,
        batch_size=500,
        remove_columns=val_df.columns.tolist()
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    manifest = {
        "loss_scope": config["LOSS_SCOPE"],
        "data_seed": config["DATA_SEED"],
        "n_dedup_removed": int(n_before - n_after),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "instruction_overlap_train_val": len(overlap),
    }

    return train_ds, val_ds, manifest
