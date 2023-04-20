# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/20 14:16:25
@author: lichunyu
'''
from transformers import AutoTokenizer

from chatglm_lora import CausalDataset


# tokenizer = AutoTokenizer.from_pretrained("/home/lichunyu/pretrain_models/chatglm-6b", trust_remote_code=True, revision="main")


dataset = CausalDataset(
    file_path="data_example.jsonl",
    tokenizer_name_or_path="/home/lichunyu/pretrain_models/chatglm-6b"
)
x = dataset[0]

...