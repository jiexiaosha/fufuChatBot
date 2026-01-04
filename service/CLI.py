from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from component.template import template_dict
import copy
import torch
from threading import Thread

def buildPrompt(tokenizer, template, history, query, system):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format 
    system = system if system is not None else template.system

    history.append({"role":"user","message":query})
    input_ids = []
    system_text = system_format.format(content = system)
    input_ids = tokenizer.encode(system_text, add_special_tokens=False)

    for convs in history:
        role, message = convs['role'], convs['message']
        if role == 'user':
            message = user_format.format(content = message, stop_token = tokenizer.eos_token)
        else:
            message = assistant_format.format(content = message, stop_token = tokenizer.eos_token)
        input_ids += tokenizer.encode(message, add_special_tokens=False) 
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids

def main():
    name = "fufu"
    max_new_tokens = 512
    top_p = 0.9
    temperature = 1.2
    repetition_penalty = 1.0
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path = "model/qwen3-4b-fufuchat-sft-qlora"
    template = "qwen3"
    system_text = "你的名字是芙，说话时自称喜欢用芙，比如‘芙喜欢吃橘子’,性格乐观，活泼好动，家里养了一只小暹罗猫，但你喜欢叫它小逻辑猫。当有人说你笨，你只会立马回一句‘揍你’而不会另外多说什么"
   
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, use_fast = True)
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eos_id
        tokenizer.bos_token_id = tokenizer.eos_id
        tokenizer.eos_token_id = tokenizer.eos_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  
        device_map="auto",           
        trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path).eval()

    history=[]

    query = input("你：")
    while True:
        query = query.strip()
        input_ids = buildPrompt(tokenizer, template_dict[template], copy.deepcopy(history), query, system = system_text).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,         
            skip_special_tokens=True  
        )

        generation_kwargs = dict(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id, streamer=streamer
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print(f"{name}：", end="", flush=True)
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print() 

        # outputs = outputs.tolist()[0][len(input_ids[0]):]
        # response = tokenizer.decode(outputs)
        response = response.strip().replace(template_dict[template].stop_word, "").strip()

        history.append({"role": "user", 'message':query})
        history.append({"role": 'assistant', 'message': response})

        # print(f"{name}：{response}")
        query = input('你：')

if __name__ == '__main__':
    main()

