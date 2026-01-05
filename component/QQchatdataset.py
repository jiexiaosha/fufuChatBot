import json
from loguru import logger
from torch.utils.data import Dataset


class QQchatDataset(Dataset):
    def __init__(self, file, tokenizer, max_length, template):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system
        logger.info(f'正在加载数据集{file}')

        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        
        logger.info(f'数据集{file}加载完成，共{len(data_list)}条数据')
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        
        data = self.data_list[idx]
        data = json.loads(data)
        input_ids, target_msk = [],[]

        system = data.get('system')
        if isinstance(system, str):
            system = system.strip()
        else:
            system = self.system

        if system:
            system_text = self.system_format.format(content=system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_msk = [0] * len(input_ids)
        
        conversations = data.get('conversations', [])

        for msg in conversations:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()

            
            if not content:
                continue

            if role == "user":
                text = self.user_format.format(content=content, stop_token=self.tokenizer.eos_token)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids.extend(tokens)
                target_msk.extend([0] * len(tokens))  # 不计算 loss

            elif role == "assistant":
                text = self.assistant_format.format(content=content, stop_token=self.tokenizer.eos_token)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids.extend(tokens)
                target_msk.extend([1] * len(tokens))  # 计算 loss

            else:
            # 忽略未知 role（如 system 在对话中重复出现）
                continue

        assert len(input_ids) == len(target_msk)
        input_ids = input_ids[:self.max_length]
        target_msk = target_msk[:self.max_length]
        attn_msk = [1] * len(input_ids)
        assert len(input_ids) == len(target_msk) == len(attn_msk)

        inputs = {
            'input_ids' : input_ids,
            'attention_mask' : attn_msk,
            'target_mask' : target_msk
        }

        return inputs
