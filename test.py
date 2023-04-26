# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/20 14:16:25
@author: lichunyu
'''
from transformers import AutoTokenizer, default_data_collator, AutoModel
from torch.utils.data import DataLoader
import torch

from chatglm_lora import CausalDataset, CausalCollator


tokenizer = AutoTokenizer.from_pretrained("/home/lichunyu/pretrain_models/chatglm-6b", trust_remote_code=True, revision="main")
# model = AutoModel.from_pretrained("/home/lichunyu/pretrain_models/chatglm-6b", trust_remote_code=True, revision="main").half().cuda()

dataset = CausalDataset(
    file_path="data.jsonl",
    tokenizer_name_or_path="/home/lichunyu/pretrain_models/chatglm-6b"
)
dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=CausalCollator())
# model.eval()
for batch in dataloader:
    batch = {k: v.cuda() for k, v in batch.items()}
    # output = model(**batch)
    ...
# x = dataloader[0]

...