from component.chat_LoRA_dataset import ChatLoraDataset
from component.template import template_dict
from typing import Any, Dict, List
from loguru import logger
from component.QQchatdataset import QQchatDataset
import torch


# 本段参考了firefly的一部分代码
def load_dataset(args, tokenizer):
    template = template_dict[args.template_name]
    if args.dataset_type == 'chat_lora':
        train_dataset = ChatLoraDataset(args.train_file, tokenizer, args.max_length, template)
    elif args.dataset_type == 'qqchat':
        train_dataset = QQchatDataset(args.train_file, tokenizer, args.max_length, template)
    return train_dataset

class data_collator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lenth = [len(x["input_ids"]) for x in batch]

        # 动态取batch长度
        batch_max_len = min(max(lenth), self.max_seq_length)

        input_ids_batch, attn_msk_batch, target_msk_batch = [], [], []

        for x in batch:
            input_ids = x['input_ids']
            attn_msk = x['attention_mask']
            target_msk = x['target_mask']
            if input_ids is None:
                logger.info("存在空input_ids")
                continue
            
            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attn_msk = attn_msk + [0] * padding_len
            target_msk = target_msk + [0] * padding_len

            input_ids_batch.append(input_ids)
            attn_msk_batch.append(attn_msk)
            target_msk_batch.append(target_msk)

    
        input_ids_batch = torch.tensor(input_ids_batch, dtype = torch.long)
        attn_msk_batch = torch.tensor(attn_msk_batch, dtype = torch.long)
        target_msk_batch = torch.tensor(target_msk_batch, dtype = torch.long)

        labels = torch.where(target_msk_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attn_msk_batch,
            'labels': labels
        }
        return inputs
