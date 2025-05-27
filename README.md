参考项目：[Jeryi-Sun/ReDEeP-ICLR](https://github.com/Jeryi-Sun/ReDEeP-ICLR)：
[](figure/1.png)

方法图示：
[](figure/2.jpg)

## 虚拟环境

```bash
conda create -n redeep python=3.9
conda activate redeep
pip install numpy==1.26.0 torch==2.0.1 accelerate==0.23.0 pandas==2.1.1 scikit-learn===1.3.1 sentence_transformers ipykernel
python -m ipykernel install --user --name redeep
jupyter kernelspec list

cd src
pip install -e transformers
```

## 项目结构

```
dataset/
    copy_heads # 复制头信息
    dolly # 数据集
    ragtruth # 数据集
    token_hyperparameter # AARF.py 的超参数

log/ # 保存运行结果

src/ # 保存项目脚本
    AARF.py
    detect.py
    regress.py

transformers/ # 保存修改的 transformers 库

test.sh # 更多研究所使用的脚本
```

## 模型

```bash
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --token Your_token --local-dir model/Llama-2-7b-chat-hf
huggingface-cli download --resume-download BAAI/bge-base-en-v1.5 --local-dir model/bge-base-en-v1.5
```

## 数据集

数据集下载：[google drive](https://drive.google.com/file/d/1s-pmaBQutC6eQGtk2F3uKaMSkn_iGQwR/view?usp=sharing)

以 RAGTruth 数据集为例，其结构为：

1. `response.jsonl`：  
```json
{
    "id": str, # 回应的索引 
    "source_id": str, 来源信息的索引,
    "model": str, # 生成回应的模型：gpt-4-0613、gpt-3.5-turbo-0613、mistral-7B-instruct、llama-2-7b-chat、llama-2-13b-chat、llama-2-70b-chat
    "temperature": float, # 生成回应的温度：0.7、0.775、1.0、0.85、0.925 
    "labels": [
        {
            "start": int, # 在回应中的起始位置
            "end": int, # 在回应中的终止位置
            "text": str, # 回应中的幻觉文本
            "meta": str, # 注释人员对幻觉的评论
            "label_type": str, # 幻觉类型
            "implicit_true": bool, # 是否和上下文冲突：回应正确，上下文没提及
            "due_to_null": bool, # 幻觉是否由 null 值引起
        },
        ...
    ], 
    "split": str, # train、test 
    "quality": str, # good（回应质量好）、incorrect_refusal（尽管存在相关上下文，模型错误地拒绝回答）、truncated（回应意外截断） 
    "response": str, 大模型对给定指令的回应
}
```  

2. `source_info.jsonl`：
```json
{
    "source_id": str, # 来源信息的索引
    "task_type": str, # Summary、QA、Data2txt 
    "source": str, # 原始内容来源：CNN/DM、Recent News、Yelp、MARCO 
    "source_info": str, # RAG 设置的基本内容：Summary是字符串，其他任务是字典
    "prompt": str, # 用来生成回应的提示
}
```

## 更多研究

在相同的评估指标下，将 Copy Heads 替换为每层的 Attention Heads，得到的结果：  
![](figure.png)
