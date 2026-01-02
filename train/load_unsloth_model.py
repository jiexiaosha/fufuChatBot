from unsloth import FastLanguageModel
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .get_linear_names import get_linear_names


def load_unsloth_model(args, training_args):
    model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.max_length,
    dtype = None,
    trust_remote_code = True,
    load_in_4bit = True
    )

    logger.info("初始化PEFT模型")

    target_modules = get_linear_names(model)
    model.gradient_checkpointing_enable()
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        lora_alpha = args.lora_alpha,
        target_modules = target_modules,
        lora_dropout = args.lora_dropout,
        bias = "none",
    )
    logger.info(f'target_modules: {target_modules}')

    return {
        'model': model,
        'ref_model': None,
        'peft_config': None
    }



