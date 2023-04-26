# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/18 16:07:43
@author: lichunyu
'''
import json


def gen_pair_data(data: list, max_text_length: int):
    result = []
    history = ""
    prompt = ""
    for i, d in enumerate(data):
        q, a = d["q"], d["a"]
        if history:
            prompt = history + "[Round {}]\n问：{}".format(i, q)
        else:
            prompt = q
        if len(prompt) > max_text_length:
            break
        result.append({
            "q": prompt,
            "a": a,
            "is_multi_turn": 1 if history else 0
        })
        history += "[Round {}]\n问：{}\n答：{}\n".format(i, q, a)
    return result


def jsonline(file_path: str):
    with open(file_path, "r") as f:
        while 1:
            if i := f.readline():
                yield json.loads(i)
            else:
                return


def get_data(file_path:str, max_text_length: int=2000):
    data = []
    for i in jsonline(file_path):
        data.extend(gen_pair_data(i, max_text_length=max_text_length))
    return data


if __name__ == "__main__":
    for i in get_data("data_example.jsonl"):
        print(i)
    ...