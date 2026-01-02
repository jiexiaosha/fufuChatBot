from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import bitsandbytes as bnb
import torch.nn as nn


def get_linear_names(model):
    linear_names = set()

    # 因为用qlora，所以直接判断为bnb.nn.Linear4bit了
    linear_class = (bnb.nn.Linear4bit)
    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split('.')
            linear_names.add(names[-1])  

    linear_names = list(linear_names)
    logger.info(f"找到模型 {model.__class__.__name__} 线性层：{linear_names} ")
    return linear_names 