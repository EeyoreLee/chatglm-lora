# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/17 10:18:56
@author: lichunyu
'''
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import T_co
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerBase
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from accelerate.state import AcceleratorState
from accelerate.utils import LoggerType
from tqdm.auto import tqdm

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
        is_multi_turn = self.data[index].get("is_multi_turn", 0)

        input_ids, labels, question = self.get_input_ids_and_labels(q, a, is_multi_turn)
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

    def get_input_ids_and_labels(self, q, a, is_multi_turn=0):
        inputs = self.tokenizer(
                q,
                add_special_tokens = True,
                truncation='longest_first',
                max_length = self.max_q_length,
                return_tensors = 'pt',
                return_attention_mask=False
        )["input_ids"]

        if is_multi_turn != 0:
            answer_prompt = self.tokenizer(
                Prompt.answer,
                add_special_tokens=True,
                return_tensors = 'pt',
                return_attention_mask=False
            )["input_ids"]
            question = torch.concat([inputs, answer_prompt], dim=-1)
        else:
            question = inputs

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

        input_ids = torch.concat([question, input_answer], dim=-1).squeeze(0)
        labels = torch.concat([torch.full((1, question.size(-1)), -100), label_answer], dim=-1).squeeze(0)
        return input_ids, labels, question

    def get_attention_mask(self, input_ids, q_length):
        attention_mask = torch.ones((input_ids.size(-1), input_ids.size(-1)))
        attention_mask.tril_()
        attention_mask[..., :q_length] = 1
        attention_mask.unsqueeze_(0)
        attention_mask = (attention_mask < 0.5).bool()  # cause GLM use masked_fill_(mask, -1000.)
        return attention_mask

    def get_position_ids(self, input_ids, q_length):
        position_ids_1 = torch.arange(input_ids.size(-1))
        position_ids_2 = torch.concat([torch.zeros(q_length), torch.arange(1, input_ids.size(-1)-q_length+1)], dim=-1)
        position_ids_2d = torch.stack([position_ids_1, position_ids_2])
        return position_ids_2d


@dataclass
class CausalCollator:

    tokenizer: PreTrainedTokenizerBase = field(default=None, metadata={"help": "tokenizer of the LLM"})

    def __call__(self, batch):
        max_length = max([i["input_ids"].size(-1) for i in batch])
        input_ids = pad_sequence(
            [i["input_ids"] for i in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 3
        ).int()
        attention_mask = torch.stack(
            [F.pad(
                    i["attention_mask"],
                    (
                        0,
                        max_length-i["attention_mask"].size(-1),
                        0,
                        max_length-i["attention_mask"].size(-1)
                    ),
                    value=False
                ) for i in batch]
            )
        position_ids = torch.stack([F.pad(i["position_ids"], (0, max_length-i["position_ids"].size(-1)),mode="replicate") for i in batch]).int()
        labels = pad_sequence([i["labels"] for i in batch], batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels
        }



def main():

    # TODO collect with a dataclass
    model_name_or_path = "/home/lichunyu/pretrain_models/chatglm-6b"  # change to `chatglm-6b` without local weights
    tokenizer_name_or_path = "/home/lichunyu/pretrain_models/chatglm-6b" # change to `chatglm-6b` without local weights
    file_path = "data_example.jsonl"
    max_q_length = 1500
    max_a_length = 500
    batch_size = 2
    learning_rate = 1e-4
    accumulate_step=1
    num_epochs=100
    num_warmup_steps=0
    # save_filename="output_dir/chatglm_lora"
    save_filename="chatglm_lora.pth"

    accelerator = Accelerator()
    # accelerator = Accelerator(log_with=[LoggerType.TENSORBOARD])
    accelerator.print(f"{AcceleratorState()}")
    hps = {}
    # accelerator.init_trackers("chatglm_lora", config=hps)

    with accelerator.main_process_first():
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query_key_value"], fan_in_fan_out=False, bias="all"
        )

        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, revision="main").half()
        model = get_peft_model(model, peft_config)

    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    train_dataset = CausalDataset(
        file_path=file_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_q_length=max_q_length,
        max_a_length=max_a_length
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=CausalCollator())

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    max_train_steps = len(train_dataloader) // accumulate_step * num_epochs
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

    for epoch in range(num_epochs):
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, disable=(not accelerator.is_local_main_process))):
        # for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            accelerator.backward(loss)
            optimizer.step()
            # accelerator.log({"training_loss": loss}, step=step)
            epoch_loss += loss.item()
        accelerator.print(f"loss: {epoch_loss/(step+1)}")

        # HACK
        # if epoch > 20 and epoch%5 == 0:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     accelerator.save(unwrapped_model.state_dict(), save_filename+f"epoch_{epoch}.pth")

    # accelerator.end_training()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), save_filename)


if __name__ == "__main__":
    main()