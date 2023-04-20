# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/17 10:18:56
@author: lichunyu
'''
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from accelerate.state import AcceleratorState

from utils import get_data
from constants import Prompt, TokenNum


class CausalDataset(Dataset):

    def __init__(
            self,
            file_path: str,
            tokenizer_name_or_path: str,
            max_q_length=1500,
            max_a_length=500,
        ) -> None:
        super().__init__()
        self.data = get_data(file_path=file_path, max_text_length=2000)  # HACK
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, revision="main")
        self.max_q_length = max_q_length
        self.max_a_length = max_a_length

    def __getitem__(self, index) -> T_co:
        q = self.data[index]["q"]
        a = self.data[index]["a"]

        input_ids, labels, question = self.get_input_ids_and_labels(q, a)
        attention_mask = self.get_attention_mask(input_ids, question.size(-1))
        position_ids = self.get_position_ids(input_ids, question.size(-1))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)

    def get_input_ids_and_labels(self, q, a):
        inputs = self.tokenizer(
                q,
                add_special_tokens = False,
                truncation='longest_first',
                max_length = self.max_q_length-TokenNum.answer,
                return_tensors = 'pt',
                return_attention_mask=False
        )["input_ids"]
        answer_prompt = self.tokenizer(
            Prompt.answer,
            add_special_tokens=True,
            return_tensors = 'pt',
            return_attention_mask=False
        )["input_ids"]

        # insight of picture 2 in GLM paper but is not suitable for open-source code !
        # input_answer = self.tokenizer(
        #     self.tokenizer.bos_token+a,
        #     add_special_tokens = False,
        #     truncation='longest_first',
        #     max_length = self.max_a_length,
        #     return_tensors = 'pt',
        #     return_attention_mask=False,
        # )["input_ids"]

        label_answer = self.tokenizer(
            a+self.tokenizer.eos_token,
            add_special_tokens = False,
            truncation='longest_first',
            max_length = self.max_a_length,
            return_tensors = 'pt',
            return_attention_mask=False,
        )["input_ids"]

        input_answer = label_answer

        question = torch.concat([inputs, answer_prompt], dim=-1)
        input_ids = torch.concat([question, input_answer], dim=-1).squeeze(0)
        labels = torch.concat([torch.full((1, question.size(-1)), -100), label_answer], dim=-1).squeeze(0)
        return input_ids, labels, question

    def get_attention_mask(self, input_ids, q_length):
        attention_mask = torch.ones((input_ids.size(-1), input_ids.size(-1)))
        attention_mask = attention_mask.tril()
        attention_mask[..., :q_length] = 1
        attention_mask = attention_mask.bool()
        return attention_mask

    def get_position_ids(self, input_ids, q_length):
        position_ids_1 = torch.arange(input_ids.size(-1))
        position_ids_2 = torch.concat([torch.zeros(q_length), torch.arange(1, input_ids.size(-1)-q_length+1)], dim=-1)
        position_ids_2d = torch.stack([position_ids_1, position_ids_2])
        return position_ids_2d


def main():

    # TODO collect with a dataclass
    model_name_or_path = "/home/lichunyu/pretrain_models/chatglm-6b"
    tokenizer_name_or_path = "/home/lichunyu/pretrain_models/chatglm-6b"
    file_path = "data_example.jsonl"
    max_q_length = 1500
    max_a_length = 500
    batch_size = 2
    learning_rate = 1e-4
    accumulate_step=1
    epoch=5
    num_warmup_steps=0

    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

    train_dataset = CausalDataset(
        file_path=file_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_q_length=max_q_length,
        max_a_length=max_a_length
    )
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