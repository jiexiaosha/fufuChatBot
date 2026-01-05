import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from loguru import logger
import torch
from trl import DPOTrainer, get_kbit_device_map
from .get_linear_names import get_linear_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model(args, training_args):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtypr = torch.float16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        )

    model_kwargs = dict(
        trust_remote_code = True,
        dtype = torch.float16,
        use_cache = False,
        device_map = get_kbit_device_map(),
        quantization_config = quantization_config,
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    logger.info(f'正在加载基底模型：{model.config.model_type}')
    logger.info(f'以qlora模式训练模型')

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)

    target_modules = get_linear_names(model)
    peft_config = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_alpha,
        target_modules = target_modules,
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    logger.info(f'模型内存占用：{model.get_memory_footprint()/1024/1024/1024}G')
    model.print_trainable_parameters()

    total_param = sum(p.numel() for p in model.parameters())
    logger.info("总共参数（包含可训练参数和不可训练参数）: %2.fM" % 1e6)

    return{
        'model':model,
        'peft_config':peft_config,
        'ref_model':None
    }
    
def load_tokenizer(args):

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code = True, use_fast = True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.eos_token_id = tokenizer.eos_token_id
    logger.info(f'正在加载tokenizer,总体大小：{tokenizer.vocab_size}')
    return tokenizer
    
    

