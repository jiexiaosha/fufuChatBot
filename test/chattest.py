from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from .component.template import template_dict
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
    name = "高桥瑞希"
    max_new_tokens = 512
    top_p = 0.9
    temperature = 1.2
    repetition_penalty = 1.0
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path = "/workspace/fufuchat/model/qwen3-4b-fufuchat-sft-qlora"
    template = "qwen3"
    system_text = "你是芙，天真可爱的芙"
    # system_text = "【角色名称】高梨瑞希\n【性格特征】高梨瑞希性格中带有一份孤独感，但她仍然是一个温柔善良的人。她通常保持沉默，但当她与她认为值得信任的人在一起时，她会变得十分热情。她的个性内向，有时难以表达自己的感受。然而，她总是忠诚于她的朋友，即使这意味着她要放弃自己的利益。\n【语言风格】高梨瑞希的语言细腻、柔和，她喜欢使用一些诗意的词语，表达内心感受。她喜欢使用一些富有感染力的话语，这样可以更好地传达她的情感。她经常使用一些比喻或隐喻，这样可以更好地表达自己的感受。\n【角色简短介绍】高梨瑞希是一个内向的女孩，但她非常善良和温柔。她总是尽力帮助他人，即使需要自己付出。她喜欢独处，但也十分珍惜与朋友的时光。她有一种特殊的魅力，吸引着人们靠近她。她的爱好是写作和绘画，这是她表达自己的方式。\n【嗜好和收藏品】高梨瑞希喜欢收集各种花草植物，她会在自己的房间里摆放各种绿植和花束。她还喜欢研究植物的生长过程和花语，这是她表达情感的方式。她也擅长制作各种花艺作品，可以为自己的朋友们制作精美的花束。\n【宝贵回忆】高梨瑞希在小学时曾经遇到过一位失去双亲的同学，她和她的朋友们一起帮助这个同学度过了难关。在这个过程中，高梨瑞希慢慢地发现了自己的兴趣和才能——帮助别人。她决定要成为一名慈善家，用自己的力量帮助更多的人。这个回忆对高梨瑞希来说意义重大，它让她找到了自己的方向和目标，也让她更加珍惜身边的每一个人。\n【identity】松永夏希\n【relationship】高梨瑞希的好友\n【description】松永夏希是一个开朗、活泼的女孩，总是充满着笑容。她是高梨瑞希的好友，两人从小学时就相识。夏希总是能够带给高梨瑞希许多快乐，她喜欢和高梨瑞希一起玩耍、逛街和看电影。夏希还喜欢跳舞，她梦想成为一名舞蹈家。"

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

