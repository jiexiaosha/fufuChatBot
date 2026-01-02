import argparse
import yaml as yml
import os
from os.path import join
from loguru import logger
from transformers import TrainingArguments, HfArgumentParser
from settings.custom_settings import CustomSettings
from transformers import set_seed
import json
from pathlib import Path
from train.components_ini import init_components
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type = str, default = 'fufuchat/training_settings.yml', help = '配置文件路径')

    args_cli = parser.parse_args()
    train_setting_file = args_cli.config
    
    with open(args_cli.config, 'r') as f:
        config = yml.safe_load(f)
    
    parser = HfArgumentParser((CustomSettings, TrainingArguments))
    # args是自己定义的，training_args是HF官方定义的
    args, training_args = parser.parse_dict(config)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # 所有训练日志都保存到一个太难翻了，改为炼一次保存一次，再炼再开新的文件
    i = 1
    while (Path(training_args.output_dir) / f"train{i}.log").exists():
        i += 1
    logger.add(join(training_args.output_dir, f'train{i}.log'))
    logger.info(f"train_args:{training_args}")

    # 从github，firefly中看到的，仔细想了想发现确实有必要保存训练参数
    train_settings = {
        **args.__dict__,
        **training_args.to_dict()
    }
    # 用yml会出现“!!python”一类的东西，放到别的电脑上可能用不了，因此这里用json保存
    # 其实主要还是因为能写注释
    with open(join(training_args.output_dir, 'train_settings.json'), 'w') as f:
        json.dump(train_settings, f, indent=4)
    set_seed(training_args.seed)
    return args, training_args

def main():
    args, training_args = parse_args()
    trainer = init_components(args, training_args)

    logger.info("开始训练")
    train_result = trainer.train()

    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
