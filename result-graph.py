import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

# ==================== 0. 终端路径交互选择 ====================
print("=" * 50)
print("请选择要评估的数据源：")
print("  [1] Groq (openai-gpt-oss-20b_envo_sweet.jsonl)")
print("  [2] SiliconFlow (openai-Qwen-Qwen2.5-7B-Instruct_envo_sweet.jsonl)")
print("  [3] 自定义路径")
print("=" * 50)

choice = input("请输入选项 (1/2/3，默认回车为 2): ").strip()

base_dir = 'results/baseline/envo_sweet'
if choice == '1':
    jsonl_file = os.path.join(base_dir, 'openai-gpt-oss-20b_envo_sweet.jsonl')
    model_tag = "Groq"
elif choice == '3':
    jsonl_file = input("请输入 .jsonl 文件的相对或绝对路径: ").strip()
    model_tag = "Custom"
else:
    # 默认选择 2
    jsonl_file = os.path.join(base_dir, 'Qwen-Qwen2.5-7B-Instruct_envo_sweet.jsonl')
    model_tag = "SiliconFlow"

if not os.path.exists(jsonl_file):
    raise FileNotFoundError(f"未找到数据文件: {jsonl_file}，请确认路径是否正确或是否已运行评测。")

print(f"\n[+] 正在加载评估数据 ({model_tag}): {jsonl_file}")

# ==================== 1. 数据加载与统一清洗 ====================
data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        line_clean = line.strip()
        if line_clean:
            data.append(json.loads(line_clean))

df = pd.DataFrame(data)

# 数值转换与异常值处理
df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
df['confidence'] = pd.to_numeric(df.get('confidence', 10.0), errors='coerce')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 提取 Token 消耗指标（如果存在）
if 'api_metrics' in df.columns:
    df['total_tokens'] = df['api_metrics'].apply(
        lambda x: x.get('total_tokens', np.nan) if isinstance(x, dict) else np.nan
    )
else:
    df['total_tokens'] = np.nan

# 过滤核心字段缺失值
df_clean = df.dropna(subset=['similarity', 'pred_relation', 'true_relation']).copy()
df_clean['is_correct'] = (df_clean['true_relation'] == df_clean['pred_relation'])

print(f"原始数据量: {len(df)} 条 | 清洗后有效数据量: {len(df_clean)} 条")
if len(df_clean) == 0:
    raise ValueError("清洗后没有有效数据，请检查 jsonl 文件的字段完整性。")

# ==================== 2. 生成基础评估看板 (Dashboard 1) ====================
print(f"\n正在生成基础看板: kroma_{model_tag.lower()}_full_dashboard.png ...")
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

# 2.1 混淆矩阵 (显式声明 labels 防止单类别时报警或结构缺失)
cm = confusion_matrix(df_clean['true_relation'], df_clean['pred_relation'], labels=[0, 1])
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes1[0, 0],
    xticklabels=['Related (0)', 'Unrelated (1)'],
    yticklabels=['Related (0)', 'Unrelated (1)']
)
axes1[0, 0].set_title(f'1. Confusion Matrix ({model_tag})')
axes1[0, 0].set_xlabel('Predicted')
axes1[0, 0].set_ylabel('Ground Truth')

# 2.2 相似度直方图 (独立柱状图 bars)
sns.histplot(
    data=df_clean, x='similarity', hue='pred_relation',
    element='bars', edgecolor='black', linewidth=1.0,
    palette='Set1', bins=25, multiple='stack', ax=axes1[0, 1]
)
axes1[0, 1].set_title(f'2. Embedding Similarity Distribution ({model_tag})')
axes1[0, 1].set_xlabel('Cosine Similarity')
axes1[0, 1].set_ylabel('Count')

# 2.3 动态累计指标收敛曲线
cum_p, cum_r, cum_f1 = [], [], []
for i in range(1, len(df_clean) + 1):
    sub = df_clean.iloc[:i]
    p, r, f1, _ = precision_recall_fscore_support(
        sub['true_relation'], sub['pred_relation'],
        average='binary', pos_label=0, zero_division=0
    )
    cum_p.append(p)
    cum_r.append(r)
    cum_f1.append(f1)

steps = list(range(1, len(df_clean) + 1))
axes1[1, 0].plot(steps, cum_p, label='Cumulative Precision', color='#1f77b4', linestyle='--')
axes1[1, 0].plot(steps, cum_r, label='Cumulative Recall', color='#2ca02c', linestyle=':')
axes1[1, 0].plot(steps, cum_f1, label='Cumulative F1-Score', color='#d62728', linewidth=2)
axes1[1, 0].set_title('3. Performance Convergence Curve')
axes1[1, 0].set_xlabel('Evaluated Samples')
axes1[1, 0].set_ylabel('Score')
axes1[1, 0].set_ylim(-0.05, 1.05)
axes1[1, 0].legend()
axes1[1, 0].grid(True, linestyle='--', alpha=0.6)

# 2.4 余弦相似度 vs. 模型置信度散点图
sns.scatterplot(
    data=df_clean, x='similarity', y='confidence',
    hue='pred_relation', style='is_correct', markers={True: 'o', False: 'X'},
    palette='Set1', s=70, ax=axes1[1, 1]
)
axes1[1, 1].set_title('4. Confidence vs. Similarity')
axes1[1, 1].set_xlabel('Cosine Similarity')
axes1[1, 1].set_ylabel('Model Confidence')
axes1[1, 1].grid(True, linestyle='--', alpha=0.6)

fig1.tight_layout()
out_fig1 = f'kroma_{model_tag.lower()}_full_dashboard.png'
fig1.savefig(out_fig1, dpi=300)
plt.close(fig1)
print(f"✓ 基础评估看板已保存至: {out_fig1}")

# ==================== 3. 生成进阶分析看板 (Dashboard 2) ====================
print(f"\n正在生成进阶看板: kroma_{model_tag.lower()}_advanced_dashboard.png ...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
unique_labels = df_clean['true_relation'].unique()

# 3.1 正负样本相似度分布对比 (显式指定 hue 规避 FutureWarning)
if len(unique_labels) > 1:
    sns.violinplot(
        data=df_clean, x='true_relation', y='similarity',
        hue='true_relation', legend=False,
        palette='Set2', inner='quartile', ax=axes2[0, 0]
    )
    sns.stripplot(
        data=df_clean, x='true_relation', y='similarity',
        color='black', alpha=0.4, jitter=0.2, ax=axes2[0, 0]
    )
else:
    sns.boxplot(
        data=df_clean, x='true_relation', y='similarity',
        hue='true_relation', legend=False,
        palette='Set2', ax=axes2[0, 0]
    )
    sns.stripplot(
        data=df_clean, x='true_relation', y='similarity',
        color='black', alpha=0.5, jitter=0.2, ax=axes2[0, 0]
    )
axes2[0, 0].set_title('1. Embedding Similarity by Ground Truth')
axes2[0, 0].set_xlabel('Ground Truth Relation')
axes2[0, 0].set_ylabel('Cosine Similarity')
axes2[0, 0].grid(True, linestyle='--', alpha=0.5)

# 3.2 ROC 曲线（单类别时自动展示占位提示，避免代码崩溃）
if len(unique_labels) > 1:
    y_binary = (df_clean['true_relation'] == 0).astype(int)
    scores = df_clean['similarity']
    fpr, tpr, _ = roc_curve(y_binary, scores)
    roc_auc = auc(fpr, tpr)

    axes2[0, 1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes2[0, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes2[0, 1].set_xlabel('False Positive Rate')
    axes2[0, 1].set_ylabel('True Positive Rate')
    axes2[0, 1].set_title('2. ROC Curve (Similarity as Classifier)')
    axes2[0, 1].legend(loc='lower right')
else:
    axes2[0, 1].text(
        0.5, 0.5, 'Requires both classes (0 & 1)\nto generate ROC Curve',
        ha='center', va='center', fontsize=12, color='gray'
    )
    axes2[0, 1].set_title('2. ROC Curve (Awaiting multi-class samples)')
axes2[0, 1].grid(True, linestyle='--', alpha=0.5)

# 3.3 置信度校准与准确率分段
bins = [0, 5, 7, 8, 9, 10]
df_clean['conf_bin'] = pd.cut(df_clean['confidence'], bins=bins, include_lowest=True)
conf_stats = df_clean.groupby('conf_bin', observed=False).apply(
    lambda g: pd.Series({
        'count': len(g),
        'accuracy': (g['true_relation'] == g['pred_relation']).mean() if len(g) > 0 else 0
    })
).reset_index()

ax_bar = axes2[1, 0]
ax_acc = ax_bar.twinx()
sns.barplot(data=conf_stats, x='conf_bin', y='count', color='#a1c9f4', ax=ax_bar, alpha=0.8)
ax_acc.plot(range(len(conf_stats)), conf_stats['accuracy'], color='#d62728', marker='o', lw=2)
ax_bar.set_title('3. Confidence Calibration & Frequency')
ax_bar.set_xlabel('Confidence Interval')
ax_bar.set_ylabel('Sample Count', color='#1f77b4')
ax_acc.set_ylabel('Accuracy', color='#d62728')
ax_acc.set_ylim(-0.05, 1.05)
ax_bar.grid(True, linestyle='--', alpha=0.5)

# 3.4 Token 消耗分布 / 置信度频数
if df_clean['total_tokens'].notna().sum() > 0:
    sns.histplot(df_clean['total_tokens'].dropna(), bins=20, kde=True, color='#2ca02c', ax=axes2[1, 1])
    axes2[1, 1].set_title('4. Token Usage Distribution')
    axes2[1, 1].set_xlabel('Total Tokens per Sample')
    axes2[1, 1].set_ylabel('Count')
else:
    sns.histplot(df_clean['confidence'], bins=10, discrete=True, color='#2ca02c', ax=axes2[1, 1])
    axes2[1, 1].set_title('4. Model Confidence Distribution')
    axes2[1, 1].set_xlabel('Confidence Score')
    axes2[1, 1].set_ylabel('Count')
axes2[1, 1].grid(True, linestyle='--', alpha=0.5)

fig2.tight_layout()
out_fig2 = f'kroma_{model_tag.lower()}_advanced_dashboard.png'
fig2.savefig(out_fig2, dpi=300)
plt.close(fig2)
print(f"✓ 进阶分析看板已保存至: {out_fig2}")

print(f"\n全部 8 个图表处理完成，已按数据源区分命名保存。")