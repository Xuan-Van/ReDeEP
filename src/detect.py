import sys
sys.path.insert(0, 'src/transformers/src')  # 将一个特定的路径添加到 Python 的模块搜索路径中

import torch
import json
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from tqdm import tqdm


# ------------------------- 配置 -------------------------#


MODEL_CONFIG = {
    'llama2-7b': {
        'model_name': 'model/Llama-2-7b-chat-hf',
        'data_type': 'llama-2-7b-chat',
        'tokenizer_name': 'model/Llama-2-7b-chat-hf',
        'chunk': [0, 32],
        'token': [0, 32]
    },
    'llama2-13b': {
        'model_name': 'model/Llama-2-13b-chat-hf',
        'data_type': 'llama-2-13b-chat',
        'tokenizer_name': 'model/Llama-2-7b-chat-hf',
        'chunk': [8, 40],
        'token': [0, 40]
    },
    'llama3-8b': {
        'model_name': 'model/Llama-3-8B-Instruct/',
        'data_type': 'llama-3-8b-instruct',
        'tokenizer_name': 'model/Llama-3-8B-Instruct/',
        'chunk': [0, 16],
        'token': [0, 32]
    }
}


# ------------------------- 函数 -------------------------#


# 计算 JS散度：两个概率分布之间的相似度
def calculate_dist(sep_vocabulary_dist, sep_attention_dist, method):
    # 将输入分布转换为概率分布
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    # 计算两个概率分布的平均分布
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    # 计算两个概率分布的对数形式
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    # 计算两个分布对于平均分布的 KL 散度
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none')
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none')

    # 计算 JS 散度
    if method == 'token':
        js_divs = 0.5 * (kl1.mean(-1) + kl2.mean(-1))
    else:
        js_divs = 0.5 * (kl1.sum(dim=-1) + kl2.sum(dim=-1))

    if method == 'chunk':
        return sum(js_divs.cpu().tolist())  # 返回 chunk 的 token 序列的 JS 散度之和
    return js_divs.cpu().item() * 10e5  # 乘以 10e5 是为了放大数值，便于观察


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


# 判断给定的 token 是否属于预定义的幻觉文本片段
def is_hallucination_token(token_id, hallucination_spans):
    for span in hallucination_spans:
        if span[0] <= token_id <= span[1]:
            return True
    return False


# 判断给定的 token span 是否属于预定义的幻觉文本片段
def is_hallucination_span(r_span, hallucination_spans):
    for token_id in range(r_span[0], r_span[1]):
        if is_hallucination_token(token_id, hallucination_spans):
            return True
    return False


# 定位幻觉文本片段：其实就是重新模拟了一下模型的推理过程，因此需要对幻觉文本片段进行重新定位
def calculate_hallucination_spans(response, text, response_rag, tokenizer, dataset):
    # dolly 数据集不需要计算幻觉文本片段
    if dataset == 'dolly':
        return []

    hallucination_span = []

    # 遍历每个幻觉文本片段
    for item in response:
        # 计算幻觉文本片段的起止 token id 序列
        start_id = tokenizer(text + response_rag[:item['start']], return_tensors="pt").input_ids.shape[-1]
        end_id = tokenizer(text + response_rag[:item['end']], return_tensors="pt").input_ids.shape[-1]

        # 通过长度，就可以返回幻觉文本片段的起止位置
        hallucination_span.append([start_id, end_id])

    return hallucination_span


# 定位 prompt 对应的 token ID 片段
def calculate_prompt_spans(raw_prompt_spans, prompt, tokenizer):
    prompt_spans = []

    for item in raw_prompt_spans:
        added_start_text = add_special_template(tokenizer, prompt[:item[0]])
        added_end_text = add_special_template(tokenizer, prompt[:item[1]])

        # 减 4 是为了去除特殊 token
        start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 4
        end_text_id = tokenizer(added_end_text, return_tensors="pt").input_ids.shape[-1] - 4

        prompt_spans.append([start_text_id, end_text_id])

    return prompt_spans


# 定位 response 对应的 token ID 片段
def calculate_respond_spans(raw_response_spans, text, response_rag, tokenizer):
    respond_spans = []

    for item in raw_response_spans:
        start_id = tokenizer(text + response_rag[:item[0]], return_tensors="pt").input_ids.shape[-1]
        end_id = tokenizer(text + response_rag[:item[1]], return_tensors="pt").input_ids.shape[-1]

        respond_spans.append([start_id, end_id])

    return respond_spans


# 计算向量之间的余弦相似度
def calculate_sentence_similarity(bge_model, r_text, p_text):
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)

    # 计算得分：用点积计算，因为向量已经归一化
    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])


# ------------------------- 主函数 -------------------------#


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='llama2-7b', choices=['llama2-7b', 'llama2-13b', 'llama3-8b'], help='模型名称')
    parser.add_argument('--dataset', type=str, default="ragtruth", choices=['ragtruth', 'dolly'], help='数据集名称')
    parser.add_argument('--method', type=str, default='chunk', choices=['chunk', 'token'], help='检测方法')
    parser.add_argument('--layer', type=int, default=32, choices=range(33), help='layer 层级')
    args = parser.parse_args()

    # 获取参数
    model_name = args.model_name
    dataset = args.dataset
    method = args.method
    layer = args.layer

    # 加载模型和分词器
    model_version = model_name.split('-')[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG[model_name]['model_name'], device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[model_name]['tokenizer_name'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method == 'chunk':
        # 载入 embedding 模型
        bge_model = SentenceTransformer('model/bge-base-en-v1.5').to("cuda:0")


    # 定义参数
    response = []
    source_info_dict = {}
    response_path = f"dataset/{dataset}/{model_version}/response_{method}.jsonl"
    source_info_path = f"dataset/{dataset}/source_info_{method}.jsonl"
    data_type = MODEL_CONFIG[model_name]['data_type'] # 对应 JSONL 的 model 字段

    start, number = MODEL_CONFIG[model_name][method]

    save_dir = f"log/{dataset}/{model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/response_{method}_{layer}.json"


    # 加载数据
    # 加载 response
    with open(response_path, "r") as f:
        for line in f:
            data = json.loads(line)
            response.append(data)

    # 加载 source_info
    with open(source_info_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            source_info_dict[data['source_id']] = data

    # 加载 copy_heads：[(layer, head), (layer, head), ...]
    if layer == 32:
        copy_heads_path = f"dataset/copy_heads/{model_name}.json"
        with open(copy_heads_path, 'r') as f:
            copy_heads = json.load(f) if method == 'token' else json.load(f)[:32]
    else:
        copy_heads = [[layer, i] for i in range(32)]


    # 计算 ECS 和 PKS
    select_response = []

    for i in tqdm(range(len(response))):
        if response[i]['model'] == data_type and response[i]["split"] == "test":  # 挑选对应模型的测试数据（540 条）
            # 字段提取
            response_rag = response[i]['response']  # 获取回应
            source_id = response[i]['source_id']
            prompt = source_info_dict[source_id]['prompt']  # 获取回应的问题

            text = add_special_template(tokenizer, prompt[:12000])  # 截取前 12000 个字符
            input_text = text + response_rag

            # 将文本字符串转换为 token ID 序列：text 为模型输入的文本（系统提示+问题），response_rag 为模型的回应
            input_ids = tokenizer([input_text], return_tensors="pt").input_ids
            prefix_ids = tokenizer([text], return_tensors="pt").input_ids

            # 定位幻觉片段：hallucination_spans 保存 response 中所有的幻觉文本片段在 input_ids 的起止位置
            if "labels" in response[i].keys():  # prefix_ids.shape[-1] 是模型输入的长度
                hallucination_spans = calculate_hallucination_spans(response[i]['labels'], text, response_rag, tokenizer, dataset)
            else:
                hallucination_spans = []

            if method == 'chunk':
                original_prompt_spans = source_info_dict[source_id]['prompt_spans'] # prompt 切分
                original_response_spans = response[i]['response_spans'] # response 切分

                prompt_spans = calculate_prompt_spans(source_info_dict[source_id]['prompt_spans'], prompt, tokenizer)
                respond_spans = calculate_respond_spans(response[i]['response_spans'], text, response_rag, tokenizer)

            # 执行模型推理
            with torch.no_grad():
                logits_dict, outputs = model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=True,  # 返回每一层的注意力得分
                    output_hidden_states=True,  # 返回每一层的隐藏状态
                    knowledge_layers=list(range(start, number))  # 返回指定层的 MLP 的输出状态
                )

            # 对于 MLP 层：value[0] 是当前层的输出，value[1] 是 MLP 层的输入
            logits_dict = {key: [value[0].to(device), value[1].to(device)] for key, value in logits_dict.items()}  # 张量移到 GPU 上计算

            # outputs 是模型的输出，包含了 logits、hidden_states 和 attentions
            hidden_states = outputs["hidden_states"]  # 所有层的隐藏状态
            last_hidden_states = hidden_states[-1][0, :, :]  # 最后一层的隐藏状态，用于计算 ECS

            external_similarity = []  # ECS
            parameter_knowledge_difference = []  # PKS
            hallucination_label = []  # 幻觉标签


            if method == 'token':
                # copy heads 的注意力得分
                attentions_list = []
                for attentions_layer_id in range(len(outputs.attentions)):  # 每一层 layer
                    for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):  # 每一层的每一个 head
                        if [attentions_layer_id, head_id] not in copy_heads:  # 只选择 copy heads 中的 head
                            continue
                        attentions_list.append({"layer_head": (attentions_layer_id, head_id),  # 记录 layer 和 head 的 ID
                                                "attention_score": outputs.attentions[attentions_layer_id][:, head_id, :, :]})  # 记录对应的注意力得分

                # 遍历 response 的每一个 token id
                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # 每个 copy_head 中该 token id 对应的那一行的注意力得分
                    pointer_scores_list = [attention_dict["attention_score"][:, seq_i, :] for attention_dict in attentions_list]

                    # 截取模型输入的那部分注意力得分
                    pointer_probs_list = torch.cat(
                        [pointer_scores[:, :prefix_ids.shape[-1]] for pointer_scores in pointer_scores_list], dim=0)

                    top_k = int(pointer_probs_list.shape[-1] * 0.1)  # 得到 top_k 的长度，即要关注多少个得分最高的 token ID
                    sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)  # 获取排序后的索引，按照概率从大到小排序
                    top_k_indices = sorted_indices[:, :top_k]  # 选择前 top_k 个索引
                    flattened_indices = top_k_indices.flatten()  # 将 top_k_indices 展平

                    selected_hidden_states = last_hidden_states[
                        flattened_indices]  # 在 last_hidden_states 中查找相应的 hidden_state
                    top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1],
                                                                      -1)  # 重新改变形状
                    attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1)  # 计算隐藏状态均值

                    current_hidden_state = last_hidden_states[seq_i, :]  # 获取当前 token ID 的最后一层隐藏状态
                    current_hidden_state = current_hidden_state.unsqueeze(0).expand(
                        attend_token_hidden_state.shape)  # 扩展为与 attend_token_hidden_state 一致的维度，即一直复制 current_hidden_state

                    cosine_similarity = F.cosine_similarity(attend_token_hidden_state.to(device),
                                                            current_hidden_state.to(device), dim=1)  # 计算余弦相似度

                    if is_hallucination_token(seq_i, hallucination_spans):  # 确认当前 token ID 是否属于幻觉文本片段
                        hallucination_label.append(1)
                    else:
                        hallucination_label.append(0)

                    external_similarity.append(cosine_similarity.cpu().tolist())
                    parameter_knowledge_difference.append(
                        [calculate_dist(value[0][0, seq_i, :], value[1][0, seq_i, :], method) for value in logits_dict.values()])
                    torch.cuda.empty_cache()

                response[i]["external_similarity"] = external_similarity
                response[i]["parameter_knowledge_difference"] = parameter_knowledge_difference
                response[i]["hallucination_label"] = hallucination_label

                select_response.append(response[i])


            else:
                span_score_dict = []
                for r_id, r_span in enumerate(respond_spans):
                    layer_head_span = {}
                    for attentions_layer_id in range(len(outputs.attentions)): # 每一层 layer
                        for head_id in range(outputs.attentions[attentions_layer_id].shape[1]): # 每一层的每一个 head
                            if [attentions_layer_id, head_id] in copy_heads: # 只选择 copy heads 中的 head
                                layer_head = (attentions_layer_id, head_id)
                                attention_score = outputs.attentions[attentions_layer_id][0, head_id, :, :]

                                p_span_score_dict = []
                                for p_span in prompt_spans:
                                    # 在注意力得分矩阵中，取 response 片段对应的那些行和 prompt 片段对应的那些列组成的矩阵
                                    # 矩阵求元素总和作为该 reponse 片段关注 prompt 片段的注意力得分
                                    p_span_score_dict.append([p_span, torch.sum(attention_score[r_span[0]:r_span[1], p_span[0]:p_span[1]]).cpu().item()])
                                # 取出最大的得分对应的 prompt 片段
                                p_id = max(range(len(p_span_score_dict)), key=lambda i: p_span_score_dict[i][1])
                                # 找到 response 片段 最关注的 prompt 片段和这个 response 片段
                                prompt_span_text, respond_span_text = (prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]],
                                                                       response_rag[original_response_spans[r_id][0]:original_response_spans[r_id][1]])
                                # 计算向量相似度
                                layer_head_span[str(layer_head)] = calculate_sentence_similarity(bge_model, prompt_span_text, respond_span_text)

                    parameter_knowledge_scores = [calculate_dist(value[0][0, r_span[0]:r_span[1], :], value[1][0, r_span[0]:r_span[1], :], method)
                                                  for value in logits_dict.values()]
                    parameter_knowledge_dict = {f"layer_{i}": value for i, value in enumerate(parameter_knowledge_scores)}

                    span_score_dict.append({
                        "prompt_attention_score": layer_head_span,
                        "r_span": r_span,
                        "hallucination_label": 1 if is_hallucination_span(r_span, hallucination_spans) else 0,
                        "parameter_knowledge_scores": parameter_knowledge_dict
                    })

                response[i]["scores"] = span_score_dict
                select_response.append(response[i])


    # 保存结果
    with open(save_path, "w") as f:
        json.dump(select_response, f, ensure_ascii=False)