from .load_model import load_tokenizer
from .load_dataset import load_dataset, data_collator
from loguru import logger
from .load_unsloth_model import load_unsloth_model
from transformers import Trainer

def init_components(args, training_args):

    logger.info("初始化组件")

    tokenizer = load_tokenizer(args)
    if args.use_unsloth:
        components = load_unsloth_model(args, training_args)
    else:
        components = load_model(args, training_args)

    model = components['model']
    ref_model = components['ref_model']
    peft_model = components['peft_config']

    logger.info(f"初始化组件完成, model: {model}\n=========正在加载数据集==========")

    train_dataset = load_dataset(args, tokenizer)
    data_collated = data_collator(tokenizer, args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collated,
    )

    return trainer


