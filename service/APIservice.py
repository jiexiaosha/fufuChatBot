import httpx
import os
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import socket
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
# from settings import Settings
from loguru import logger
import uuid
import time
from component.template import template_dict
from transformers import  (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from peft import PeftModel
import torch
from threading import Thread
import copy

TEMPLATE_NAME = "qwen3"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "fufuchat/model/qwen3-4b-fufuchat-sft-qlora"
SYSTEM  = "你的名字是芙，说话时自称喜欢用芙，比如‘芙喜欢吃橘子’,性格乐观，活泼好动，家里养了一只小暹罗猫，但你喜欢叫它小逻辑猫。当有人说你笨，你只会立马回一句‘揍你’而不会另外多说什么"
NAME = "FUFU"

model = None
tokenizer = None

template = template_dict[TEMPLATE_NAME]

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []  # [{"role": "user", "message": "..."}, ...]

class ChatResponse(BaseModel):
    response: str

def buildPrompt(tokenizer, template, history, query, system):
    logger.info(f"用户：{query}")
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format 
    system = system if system is not None else template.system

    hist = copy.deepcopy(history)
    hist.append({"role":"user","message":query})
    input_ids = []
    system_text = system_format.format(content = system)
    input_ids = tokenizer.encode(system_text, add_special_tokens=False)

    for convs in hist:
        role, message = convs['role'], convs['message']
        if role == 'user':
            message = user_format.format(content = message, stop_token = tokenizer.eos_token)
        else:
            message = assistant_format.format(content = message, stop_token = tokenizer.eos_token)
        input_ids += tokenizer.encode(message, add_special_tokens=False) 
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    logger.info("加载模型与tokenizer中")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,  
        device_map="auto",           
        trust_remote_code=True
        )

    model = PeftModel.from_pretrained(model, ADAPTER).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, 
        trust_remote_code = True, 
        use_fast = True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("正在启动fufuchat")
    yield
    logger.info("正在关闭fufuchat")

app = FastAPI(title = "fufu character-roleplay chat bot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/fufuchat")
async def fufuChat(request: ChatRequest):
    global model,tokenizer,template
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    query = request.query

    try:
        query = query.strip()


        inputs = buildPrompt(
            tokenizer, 
            template, 
            copy.deepcopy(request.history), 
            request.query, 
            system = SYSTEM
            ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,         
            skip_special_tokens=True,
            timeout = 20.0  
        )

        generation_kwargs = dict(
            input_ids=inputs,
            max_new_tokens=512,
            do_sample=True,
            streamer=streamer,
            top_p=0.9,
            temperature=1.2,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id
            )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        response = []
        async def event_generator():
            for text in streamer:
                response.append(text)
                yield f"{text}"

            complete_answer = "".join(response).strip()
            logger.info(f"芙芙：{complete_answer}")

            yield "data: [DONE]" 

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"生成出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
