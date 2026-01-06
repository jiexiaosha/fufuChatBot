# FuFuChat 项目说明

## 写在最开始

本项目的路径引用可能较为混乱，如果无法运行本项目请在issue中提供报错，我会尽快修复     
gitee不支持免费版用户上传超过100M的文件，请前往蓝奏云下载以下文件并解压到model路径中，确保文件夹名称为qwen3-4b-fufuchat-sft-qlora
```bash
https://www.ilanzou.com/s/i3gn7MXn
```  
同时请前往run_demo.bat中补全第三十一行的HF_TOKEN

## 快速开始

- **运行演示**：双击 `run_demo.bat` 即可启动本地 CLI 对话演示。
- **自定义角色风格**：如需调整 AI 的角色设定，请编辑 `service/CLI/system.txt` 文件中的系统提示（system prompt）。
- **API 接入**：若需通过 API 调用服务，请请确保当前IDE命令行在项目根目录，并运行service.APIservice  
初次使用时会下载约12G的基础模型

```bash
uvicorn service/APIservice:fufuchat --host 0.0.0.0 --port 8000
```

- **模型训练**：如需微调模型，请在 IDE 中执行以下命令：

```bash
python start_train.py --config fufuchat/settings/training_settings-template.yml
```

>📌 **提示**：  
> 配置文件 `fufuchat/settings/training_settings-template.yml` 中对关键训练参数有详细说明。可按照其中说明对重要参数进行调整。

### 支持的数据集格式

当前仅支持如下 JSON 格式：

```json
{
  "system": "",
  "conversations": [
    {"user": "你好！", "assistant": "嗨～"},
    {"user": "今天过得怎么样？", "assistant": "还不错，谢谢关心！"}
  ]
}
```

> 💡 数据集目前来源于 Hugging Face，但由于角色同质化严重，现有模型的对话表现尚不理想。高质量、多样化的角色数据正在构建中，到时候会跟 **FuFuChat-v2**一起放出来

---

## 开发者笔记（TODO & 想法）
**TODO**：假如用在多人聊天中的话,应该可以考虑改下mask的遮掩方式，就我自身而言，最多回复前30条的消息，我给依次添加mask就好了    
应该可以拿群聊天记录，然后识别每个人的独特名字并替换为user1,user2的模板，这样AI有可能会学会@人    
也许替换为随机生成的字母加数字更好点，能让AI学习到@实际存在的人而不是@user1这种   

### 2026.1.2再编:
> 看了这么多论文，感觉还是得从数据集构建入手，loss值的计算方式，mask的遮掩模式，再怎么仔细去改还不如力大砖飞  而且能够模仿一个人的说话方式也够了，等需要模仿另外一个人的时候再炼一个adapter就行    

-  **TODO**：那我还得写个adapter选择器

### 2026.1.5:
- 行不通，一是很多人聊天的时候喜欢把一段完整的话分成好几段发出去，要想拼合有难度，原样使用就注定训练集质量不高，二是群聊太混乱了，聊游戏那得先有游戏的背景知识，聊音乐那也得有音乐的背景知识
---

觉得这个项目有意思的人可以给我加一个小星星✨吗，秋梨膏