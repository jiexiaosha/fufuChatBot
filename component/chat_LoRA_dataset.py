import json
from loguru import logger
from torch.utils.data import Dataset

# 其实是QLoRA，但我懒得改了
class ChatLoraDataset(Dataset):
    def __init__(self, file, tokenizer, max_length, template):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system
        self.max_length = max_length
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
        
        conversations = data['conversations']

        if len(conversations) % 2 != 0:
            conversations = conversations[:-1]

        # 因为是个对话，一定记得检查数据集conversation内容为偶数条
        for i, conv in enumerate(conversations):

            # if len(conversations) % 2 != 0:
            #     continue

            human = conv['user'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content = human, stop_token = self.tokenizer.eos_token)
            # 犹豫再三还是把这个eostoken给加上了，反正可能会有其他模板，到时候也不用多改一个
            assistant = self.assistant_format.format(content = assistant, stop_token = self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens = False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens = False)

            input_ids += input_tokens + output_tokens
            target_msk += [0] * len(input_tokens) + [1] * len(output_tokens)

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
