import sys
sys.path.insert(0, 'src/transformers/src')  # 将一个特定的路径添加到 Python 的模块搜索路径中

import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------- 配置 -------------------------#


MODEL_CONFIG = {
    'llama2-7b': {
        'model_name': 'model/Llama-2-7b-chat-hf',
        'data_type': 'llama-2-7b-chat',
        'tokenizer_name': 'model/Llama-2-7b-chat-hf'
    },
    'llama2-13b': {
        'model_name': 'model/Llama-2-13b-chat-hf',
        'data_type': 'llama-2-13b-chat',
        'tokenizer_name': 'model/Llama-2-7b-chat-hf'
    },
    'llama3-8b': {
        'model_name': 'model/Llama-3-8B-Instruct/',
        'data_type': 'llama-3-8b-instruct',
        'tokenizer_name': 'model/Llama-3-8B-Instruct/'
    }
}


# ------------------------- 函数 -------------------------#


# 构造模型输入
def add_special_template(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,  # 将 messages 转换为结构化文本字符串
        tokenize=False,  # 不进行分词
        add_generation_prompt=True  # 添加生成提示的 token
    )
    return text


# ------------------------- 主函数 -------------------------#


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='llama2-7b', choices=['llama2-7b', 'llama2-13b', 'llama3-8b'], help='模型名称')
    parser.add_argument('--dataset', type=str, default="ragtruth", choices=['ragtruth', 'dolly'], help='数据集名称')
    parser.add_argument('--AARF', action='store_true', help='是否使用AARF')
    args = parser.parse_args()


    # 获取参数
    model_name = args.model_name
    dataset = args.dataset
    AARF = args.AARF
    model_version = model_name.split('-')[0]


    source_info_path = f"dataset/{dataset}/source_info_token.jsonl"
    response_path = f"dataset/{dataset}/{model_version}/response_token.jsonl"
    save_path = f"dataset/token_hyperparameter/{dataset}/{model_name}.json"


    # 加载 source_info
    source_info_dict = {}
    with open(source_info_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            source_info_dict[data['source_id']] = data


    source_id_list = []
    with open(response_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["split"] == "test":
                source_id_list.append(data["source_id"])


    test_datas_dict = {}
    source_id_set = sorted(list(set(source_id_list)))
    for item in source_id_set:
        test_datas_dict[item] = source_info_dict[item]


    with open(save_path, "r") as f:
        hypter_parameter = json.load(f)

    select_layers = hypter_parameter["select_layers"]
    select_heads = hypter_parameter["select_heads"]
    layers_max_min = hypter_parameter["layers_max_min"]
    head_max_min  = hypter_parameter["head_max_min"]
    weight = hypter_parameter["weight"]
    final_max_min = hypter_parameter["final_max_min"]


    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[model_name]['tokenizer_name'])

    if args.AARF:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG[model_name]['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            select_layers=select_layers,
            select_heads=select_heads,
            layers_max_min=layers_max_min,
            head_max_min=head_max_min,
            weight=weight,
            final_max_min=final_max_min
        )
        model.add_attention_weight = 1.2
        model.reduce_ffn_weight = 0.8
        model.threshold = 0.6

    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG[model_name]['model_name'],
            torch_dtype=torch.float16,
            device_map="auto"
        )


    final_datas = []
    for key, prompt in tqdm(test_datas_dict.items()):
        text = add_special_template(tokenizer, prompt["prompt"][:8000])
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

        if model_name == "llama3-8b":
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            outputs = model.generate(
                input_ids,
                eos_token_id=terminators,
                pad_token_id=0,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=1024
            )
        else:
            outputs = model.generate(
                input_ids,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=1024
            )

        response = outputs[0][input_ids.shape[-1]:]
        result = tokenizer.decode(response, skip_special_tokens=True)
        print(result)
        final_datas.append({"id":key, "prompt":prompt["prompt"], "response":result})


    if args.AARF:
        with open(f"log/{dataset}/{model_name}/AARF_add_{model.add_attention_weight}_reduce_{model.reduce_ffn_weight}_threshold_{model.threshold}.json", "w") as f:
            json.dump(final_datas, f, indent=4, ensure_ascii=False)

    else:
        with open(f"log/{dataset}/{model_name}/AARF_None.json", "w") as f:
            json.dump(final_datas, f, indent=4, ensure_ascii=False)