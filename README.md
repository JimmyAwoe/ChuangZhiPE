# Prompt Engineering for Code Generation

> 🎓 创智学院博士招生 - 提示词工程考核项目

---

## 📋 项目简介

本项目是**创智学院博士招生提示词工程考核**的完整解决方案。核心挑战是在严格约束下（`max_turns <= 3`），通过精心设计的提示词策略，最大化大语言模型（Qwen3-8B）在代码题求解任务上的 **pass@1** 指标。

**核心问题**：如何在有限的对话轮次内，让 LLM 生成能够通过所有测试用例的高质量代码？

---

## 🎯 核心成果

| 指标 | 结果 |
|:---:|:---:|
| **Pass@1** | **0.52** |
| 基线 Pass@1 | 0.40 |
| **相对提升** | **+30%** |
| 模型 | Qwen3-8B |
| 数据集 | 20 道竞赛编程题 |

**10/20 题目达到 100% 通过率**：Q3, Q4, Q6, Q7, Q9, Q10, Q12, Q14, Q15, Q16

---

## 🧠 解决方案架构

### 整体策略：生成-验证-修复闭环

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  多候选生成      │ ──► │   执行验证       │ ──► │   错误修复       │
│  (2 candidates) │     │  (execute_code) │     │  (error-aware)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                                               │
         └───────────────────────────────────────────────┘
                         (失败时迭代，最多 3 轮)
```

### 三轮对话策略

| 轮次 | 策略 | 候选数 | 核心设计 |
|:---:|:---|:---:|:---|
| **Turn 1** | 首次求解 | 2 | 要求两种不同算法思路，增加多样性 |
| **Turn 2** | 错误修复 | 2 | 携带上下文，根据错误类型提供针对性提示 |
| **Turn 3** | 最终修复 | 2 | 简化上下文，提示"考虑换算法" |

### 关键创新点

#### 1️⃣ 错误类型导向修复

根据执行反馈自动识别错误类型，提供差异化修复策略：

```python
def detect_error_type(feedback: str) -> str:
    # TLE (超时): 推荐 KMP、Fenwick树、二分查找等优化算法
    # WA (答案错): 检查边界条件、索引偏移、逆映射关系
    # RE (运行时错): 检查数组越界、输入解析、递归深度
    # SyntaxError: 检查缩进和括号匹配
```

#### 2️⃣ 逆映射提示（关键突破）

针对"输出索引 ≠ 输入索引"类问题，在提示中加入：

> **"BEFORE CODING: carefully identify what the OUTPUT index represents (it may differ from the input index — build reverse mappings if needed)"**

**效果**：Q4 (Bib Numbers) 的 pass@1 从 **0.00** 提升至 **1.00**

#### 3️⃣ Token 预算管理

- 动态裁剪题目、反馈和历史代码
- 头尾保留策略，防止关键信息丢失
- 预算：单次调用最大 14500 tokens

#### 4️⃣ 候选优先级排序

基于错误类型设定修复优先级：

```
TLE (4) > WA (3) > RE (2) > Other (1) > SyntaxError (0)
```

优先修复可通过算法优化解决的 TLE 问题。

---

## 📁 项目结构

```
.
├── code/
│   ├── solution.py          # 核心解决方案
│   ├── run.py               # 官方评测脚本
│   ├── llm_client.py        # LLM 调用客户端
│   └── data/
│       └── dev.jsonl        # 评测数据集（20 道题目）
├── 探索报告.md               # 详细探索报告
└── README.md                # 本文件
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd <repo-name>

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动模型服务（vLLM）

本方案使用 Qwen3-8B 模型，通过 vLLM 提供 OpenAI 兼容的 API 服务。

```bash
# 启动 vLLM 服务（需要至少 16GB GPU 显存）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8001 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1
```

服务启动后，将在 `http://localhost:8001/v1` 提供 API 接口。

### 3. 运行评测

```bash
cd code

# 完整评测（复现论文结果）
python3 run.py \
    --all \
    --run-name v6_final \
    --samples 5 \
    --workers 20 \
    --max-turns 3

# 快速测试单题
python3 run.py \
    --question 4 \
    --run-name test_q4 \
    --samples 1 \
    --max-turns 3
```

### 4. 查看结果

```bash
# 查看评测结果
ls -la outputs/v6_final/

# 查看汇总报告
cat outputs/v6_final/summary.json
```

### 参数说明

| 参数 | 说明 |
|:---|:---|
| `--all` | 评测所有题目 |
| `--question` | 评测指定题号（0-19） |
| `--run-name` | 本次运行名称（结果保存目录） |
| `--samples` | 每题采样次数（多次采样计算 pass@1） |
| `--workers` | 并行工作进程数 |
| `--max-turns` | 最大对话轮次（考核约束：<= 3） |

---

## 📊 实验迭代历程

### 第一轮迭代（V1 ~ V4）：基线建立

| 版本 | 策略 | Pass@1 | 主要发现 |
|:---:|:---|:---:|:---|
| V1 | 单轮多候选 + 执行筛选 | 0.40 | 单候选风险高 |
| V2 | 失败反馈修复 | 0.45 | 修复提示需更精准 |
| V4 | 4 候选 + Token 预算 | 0.40 | 候选过多导致截断 |

**关键教训**：候选数量需与 Token 限制平衡。

### 第二轮迭代（V5 ~ V6）：深度优化

| 版本 | 策略 | Pass@1 | 结果 |
|:---:|:---|:---:|:---|
| **V6** | 2 候选 + 错误分类 + 逆映射提示 | **0.50~0.52** | **稳定最佳** |
| V6 + I/O 修复 | 增加 I/O 规范提示 | 0.52 | 最终版本 |

**失败的尝试**：
- V7: 加入 KMP/Z-function 算法名 → 性能回归
- V8: 强制推荐 `sys.stdin.read().split()` → Q4 回归至 0.00

> **核心教训**：简洁清晰的通用指引比详尽的特定算法指导更稳健。

---

## 🔍 失败题型分析

| 题号 | 题目类型 | 失败原因 |
|:---:|:---|:---|
| Q0, Q1, Q2, Q5, Q8, Q18 | 复杂算法题 | 需要持久化线段树、复杂 DP 优化等超纲算法 |
| Q17 | 博弈论 | 胜负条件复杂，无简单公式可循 |
| Q19 | 特殊 I/O 格式 | LeetCode 风格与实际测试冲突，需 `ast.literal_eval` 解析 |
| Q11, Q13 | 偶发失败 | 模型实现不稳定 |

**结论**：约 50% 的题目需要模型知识边界外的复杂算法，提示词工程难以突破。

---

## 📖 详细报告

完整的技术细节、实验过程和方法论总结请参见 [探索报告.md](./探索报告.md)。

---

## 🛠️ 核心代码片段

### System Prompt 设计

```python
system_prompt = (
    "You are an expert competitive programmer.\n"
    "Key guidelines:\n"
    "- Always use `import sys; input = sys.stdin.readline` for fast I/O\n"
    "- Choose algorithm by constraint size: N≤10^3 → O(N²); "
    "N≤10^5 → O(N log N); N≤10^6 → O(N)\n"
    "- CRITICAL: The output index may differ from input index — "
    "build reverse mapping if needed\n"
    "- Output complete, runnable Python in ```python code blocks"
)
```

### 错误类型检测

```python
def detect_error_type(feedback: str) -> str:
    fb = feedback.lower()
    if any(k in fb for k in ["time limit exceeded", "tle"]):
        return "TLE: 推荐 KMP、Fenwick树、二分查找等优化算法..."
    if any(k in fb for k in ["wrong answer", "expected"]):
        return "WA: 检查边界条件、索引偏移、逆映射关系..."
    if any(k in fb for k in ["runtime error", "traceback"]):
        return "RE: 检查数组越界、输入解析、递归深度..."
```

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。


