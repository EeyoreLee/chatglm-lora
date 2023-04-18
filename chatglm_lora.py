# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/17 10:18:56
@author: lichunyu
'''
import logging

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torch.optim import AdamW
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from accelerate.state import AcceleratorState


class CausalDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> T_co:
        return torch.Tensor([1,2,3,4])

    def __len__(self):
        return 128


def main():

    # TODO collect with a dataclass
    model_name_or_path = "/home/lichunyu/pretrain_models/chatglm-6b"
    batch_size = 32
    learning_rate = 1e-4
    accumulate_step=1
    epoch=5
    num_warmup_steps=0

    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

    train_dataset = CausalDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    accelerator.wait_for_everyone()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query_key_value"], fan_in_fan_out=False
    )

    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, revision="main")
    model = get_peft_model(model, peft_config)

    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    max_train_steps = len(train_dataloader)//accumulate_step*epoch
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)


if __name__ == "__main__":
    main()