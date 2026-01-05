from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CustomSettings():
    model_name: str = field(metadata={"help": "基底模型路径"})
    max_length: int = field(default=256,metadata={"help": "输入最大长度"})
    train_file: str = field(default=None,metadata={"help": "训练集的路径"})
    template_name: str = field(default="", metadata={"help": "sft的数据格式"})
    lora_rank: int = field(default=16, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    use_unsloth: bool = field(default=False, metadata={"help": "use sloth or not"})
    dataset_type: str = field(default="chat_lora", metadata={"help": "dataset type"})
