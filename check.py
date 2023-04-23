# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/23 11:45:00
@author: lichunyu
'''
from transformers import AutoModel
import torch


# model = AutoModel.from_pretrained("/home/lichunyu/pretrain_models/chatglm-6b", trust_remote_code=True, revision="main").half().cuda()
# state_dict = model.state_dict()

total_nan_count = 0

for i in range(1, 9):
    state_dict = torch.load(f"/home/lichunyu/pretrain_models/chatglm-6b/pytorch_model-0000{i}-of-00008.bin", map_location="cpu")
    origin_state_dict = torch.load(f"/home/lichunyu/pretrain_models/tmp/pytorch_model-0000{i}-of-00008.bin", map_location="cpu")


    for k, v in state_dict.items():
        if (count := torch.isnan(v.half()).any().sum()) >= 0:
            total_nan_count += count
            # print(f"{k} has {count} nan.")
        if not (origin_state_dict[k] == v).all():
            print(f"`{k}` in pytorch_model-0000{i}-of-00008.bin is changed")

print(f"total nan count: {total_nan_count}")
...