import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import (
        LoraConfig,
        PromptEncoderConfig,
        TaskType,
        get_peft_model,
    )
except ImportError as exc:
    raise SystemExit(
        "peft is required. Install it with: "
        "python -m pip install peft==0.7.1"
    ) from exc


# Current stage: fine-tuning.
# This script provides small, inspectable LoRA, P-tuning, and guarded full
# fine-tuning flows for the already downloaded local MiniCPM model.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "finetune_data" / "train.jsonl"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "finetune_outputs"


class InstructionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.records = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.records[index]
        prompt = f"用户：{item['instruction']}\n助手："
        answer = item["output"] + (self.tokenizer.eos_token or "")

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collate_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    result: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        result["input_ids"].append(
            torch.cat(
                [item["input_ids"], torch.full((pad_len,), pad_token_id)]
            )
        )
        result["attention_mask"].append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        result["labels"].append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )

    return {key: torch.stack(value) for key, value in result.items()}


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def apply_method(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.method == "lora":
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.target_modules.split(","),
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        return get_peft_model(model, config)

    if args.method == "ptuning":
        config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.prompt_encoder_hidden_size,
        )
        return get_peft_model(model, config)

    if args.method == "full" and not args.dry_run and not args.allow_full_train:
        raise SystemExit(
            "Full fine-tuning updates all model parameters and is expensive. "
            "Use --allow-full-train only after you understand the GPU memory cost."
        )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM fine-tuning demo")
    parser.add_argument("--method", choices=["lora", "ptuning", "full"], default="lora")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-full-train", action="store_true")
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated MiniCPM module names for LoRA.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--prompt-encoder-hidden-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = InstructionDataset(args.data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    model = apply_method(model, args)
    model.to(device)

    total, trainable = count_parameters(model)
    print(f"method: {args.method}")
    print(f"device: {device}")
    print(f"records: {len(dataset)}")
    print(f"total_params: {total:,}")
    print(f"trainable_params: {trainable:,}")
    print(f"trainable_ratio: {trainable / total:.6f}")

    first_batch = next(iter(loader))
    print(f"sample_shape: {tuple(first_batch['input_ids'].shape)}")
    if args.dry_run:
        print("dry_run: true, no weights were updated")
        return

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
    )

    step = 0
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step: {step}, loss: {float(loss.detach().cpu()):.4f}")
            if step >= args.max_steps:
                break

    output_dir = args.output_root / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
