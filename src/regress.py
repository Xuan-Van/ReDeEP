import pandas as pd
import json
import argparse
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# ------------------------- 配置 -------------------------#


PARAMETER_CONFIG = {
    'llama2-7b': {
        'ragtruth': {
            'chunk': [3, 4, 0.6, 1],
            'token': [1, 10, 0.2, 1]
        },
        'dolly': {
            'chunk': [7, 3, 1.6, 1],
            'token': [4, 3, 0.2, 1]
        }
    },
    'llama2-13b': {
        'ragtruth': {
            'chunk': [9, 3, 1.8, 1],
            'token': [2, 17, 0.6, 1]
        },
        'dolly': {
            'chunk': [11, 3, 0.2, 1],
            'token': [4, 5, 0.6, 1]
        }
    },
    'llama3-8b': {
        'ragtruth': {
            'chunk': [2, 5, 1.2, 1],
            'token': [3, 30, 0.4, 1]
        },
        'dolly': {
            'chunk': [1, 1, 0.1, 1],
            'token': [1, 1, 0.1, 1]
        }
    }
}


# ------------------------- 函数 -------------------------#


# 构建数据集
def construct_dataframe(file_path, number, method):
    # 读取数据集
    with open(file_path, "r") as f:
        response = json.load(f)

    # 定义表头：索引、ECS、PKS、幻觉标签
    data_dict = {
        "identifier": [],
        **{f"external_similarity_{k}": [] for k in range(number)},
        **{f"parameter_knowledge_difference_{k}": [] for k in range(number)},
        "hallucination_label": []
    }

    # response_i 表示第 i 个 response，item_j 表示第 j 个 token ID
    for i, resp in enumerate(response):  # 遍历每个 response
        if resp["split"] != "test":
            continue

        if method == 'token':
            for j in range(len(resp["external_similarity"])):  # 遍历每个 token ID
                data_dict["identifier"].append(f"response_{i}_item_{j}")
                for k in range(number):  # 遍历每个具体的得分
                    data_dict[f"external_similarity_{k}"].append(resp["external_similarity"][j][k])
                    data_dict[f"parameter_knowledge_difference_{k}"].append(resp["parameter_knowledge_difference"][j][k])
                data_dict["hallucination_label"].append(resp["hallucination_label"][j])

        else:
            for j in range(len(resp["scores"])):
                data_dict["identifier"].append(f"response_{i}_item_{j}")
                for k in range(number):
                    data_dict[f"external_similarity_{k}"].append(list(resp["scores"][j]["prompt_attention_score"].values())[k])
                    data_dict[f"parameter_knowledge_difference_{k}"].append(list(resp["scores"][j]["parameter_knowledge_scores"].values())[k])
                data_dict["hallucination_label"].append(resp["scores"][j]["hallucination_label"])

    df = pd.DataFrame(data_dict)  # 转换为 DataFrame
    return df


# 计算 ECS、PKS 的 AUC（曲线下面积）和 Pearson 相关系数（PCC）
def calculate_auc_pcc(df, number):
    auc_external_similarity = []
    pearson_external_similarity = []

    auc_parameter_knowledge_difference = []
    pearson_parameter_knowledge_difference = []

    for k in range(number):
        # ECS 和幻觉标签的 AUC 和 PCC：负相关
        auc_ext = roc_auc_score(1 - df['hallucination_label'], df[f'external_similarity_{k}'])
        pearson_ext, _ = pearsonr(df[f'external_similarity_{k}'], 1 - df['hallucination_label'])
        auc_external_similarity.append((auc_ext, f'external_similarity_{k}'))
        pearson_external_similarity.append((pearson_ext, f'external_similarity_{k}'))

        # PKS 和幻觉标签的 AUC 和 PCC：正相关
        auc_param = roc_auc_score(df['hallucination_label'], df[f'parameter_knowledge_difference_{k}'])
        if df[f'parameter_knowledge_difference_{k}'].nunique() == 1:  # 检查 PKS 某一列是否所有值都相同
            print(k)
        pearson_param, _ = pearsonr(df[f'parameter_knowledge_difference_{k}'], df['hallucination_label'])
        auc_parameter_knowledge_difference.append((auc_param, f'parameter_knowledge_difference_{k}'))
        pearson_parameter_knowledge_difference.append((pearson_param, f'parameter_knowledge_difference_{k}'))

    return auc_external_similarity, auc_parameter_knowledge_difference


# 计算 response 的 AUC 和 PCC
def calculate_auc_pcc_32_32(df, top_n, top_k, alpha, auc_external_similarity, auc_parameter_knowledge_difference, m=1):
    # 选择 ECS 的 AUC 分数最高的 top_n 个特征
    top_auc_external_similarity = sorted(auc_external_similarity, reverse=True)[:top_n]

    # 选择 PKS 的 AUC 分数最高的 top_k 个特征
    top_auc_parameter_knowledge_difference = sorted(auc_parameter_knowledge_difference, reverse=True)[:top_k]

    # 对于选择好的特征，求其对应 df 列的和，表示为 ECS 和 PKS 的和
    df['external_similarity_sum'] = df[[col for _, col in top_auc_external_similarity]].sum(axis=1)
    df['parameter_knowledge_difference_sum'] = df[[col for _, col in top_auc_parameter_knowledge_difference]].sum(axis=1)

    # 计算 ECS、PKS 和的 AUC
    final_auc_external_similarity = roc_auc_score(1 - df['hallucination_label'], df['external_similarity_sum'])
    final_auc_parameter_knowledge_difference = roc_auc_score(df['hallucination_label'], df['parameter_knowledge_difference_sum'])

    # 计算 ECS、PKS 和的 PCC
    final_pearson_external_similarity, _ = pearsonr(df['external_similarity_sum'], 1 - df['hallucination_label'])
    final_pearson_parameter_knowledge_difference, _ = pearsonr(df['parameter_knowledge_difference_sum'], df['hallucination_label'])

    # 存放结果
    results = {
        f"Top {top_n} AUC External Similarity": final_auc_external_similarity,
        f"Top {top_k} AUC Parameter Knowledge Difference": final_auc_parameter_knowledge_difference,
        f"Top {top_n} Pearson Correlation External Similarity": final_pearson_external_similarity,
        f"Top {top_k} Pearson Correlation Parameter Knowledge Difference": final_pearson_parameter_knowledge_difference
    }

    # 最小最大归一化
    scaler = MinMaxScaler()

    # 归一化 ECS 和的列、PCS 和的列
    df['external_similarity_sum_normalized'] = scaler.fit_transform(df[['external_similarity_sum']])
    df['parameter_knowledge_difference_sum_normalized'] = scaler.fit_transform(df[['parameter_knowledge_difference_sum']])

    # 线性拟合 ECS 和 PKS 为 difference_normalized
    df['difference_normalized'] = m * df['parameter_knowledge_difference_sum_normalized'] - alpha * df['external_similarity_sum_normalized']

    # 计算 difference_normalized 的 AUC 和 PCC
    auc_difference_normalized = roc_auc_score(df['hallucination_label'], df['difference_normalized'])
    person_difference_normalized, _ = pearsonr(df['hallucination_label'], df['difference_normalized'])
    results.update({"Normalized Difference AUC": auc_difference_normalized})
    results.update({"Normalized Difference Pearson Correlation": person_difference_normalized})

    # 将 token 级别的预测结果转换为 response 级别的评估
    df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')  # 只区分 response，忽略 token
    grouped_df = df.groupby('response_group').agg(  # 按 response_group 分组，对每组内的数据计算聚合统计值
        difference_normalized_mean=('difference_normalized', 'mean'),  # 计算 difference_normalized 的均值
        hallucination_label=('hallucination_label', 'max')  # 有一个幻觉 token 就表明是幻觉 response
    ).reset_index()

    # 进行归一化
    min_val = grouped_df['difference_normalized_mean'].min()
    max_val = grouped_df['difference_normalized_mean'].max()
    grouped_df['difference_normalized_mean_norm'] = (grouped_df['difference_normalized_mean'] - min_val) / (max_val - min_val)

    # 计算 response 的 AUC 和 PCC
    auc_difference_normalized = roc_auc_score(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])
    person_difference_normalized, _ = pearsonr(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])

    results.update({"Grouped means AUC": auc_difference_normalized})
    results.update({"Grouped means Pearson Correlation": person_difference_normalized})

    print(json.dumps(results, ensure_ascii=False, indent=4))
    return auc_difference_normalized, person_difference_normalized


# ------------------------- 主函数 -------------------------#


if __name__ == '__main__':
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


    data_path = f"log/{dataset}/{model_name}/response_{method}_{layer}.json"
    save_path = f"log/{dataset}/{model_name}/ReDeEP_{method}_{layer}.json"


    # 加载 copy_heads
    if method == 'token':
        if layer == 32:
            copy_heads_path = f"dataset/copy_heads/{model_name}.json"
            with open(copy_heads_path, 'r') as f:
                copy_heads = json.load(f)
            sorted_copy_heads = sorted(copy_heads, key=lambda x: (x[0], x[1]))  # 按 layer 层和注意力头升序排序
        else:
            sorted_copy_heads = [[layer, i] for i in range(32)]


    # 构建数据集
    df = construct_dataframe(data_path, 32, method)


    # 计算 ECS、PKS 各自和幻觉标签的 AUC
    auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df, 32)


    # i：AUC 最高的前 i 个 ECS；j：AUC 最高的前 j 个 PKS
    # k：ECS 的权重系数 alpha；m：PKS 的权重系数
    i, j, k, m = PARAMETER_CONFIG[model_name][dataset][method]

    # 计算 response 的 AUC 和 PCC
    auc_difference_normalized, person_difference_normalized = calculate_auc_pcc_32_32(df, i, j, k, auc_external_similarity, auc_parameter_knowledge_difference, m)

    result_dict = {"auc": auc_difference_normalized, "pcc": person_difference_normalized}

    with open(save_path, 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False)