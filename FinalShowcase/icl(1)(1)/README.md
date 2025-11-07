# ICL提示策略对比系统

基于上下文学习（In-Context Learning）研究不同提示策略在分类、抽取、推理等任务上的效果与代价（准确率、成本、时延），并使用DSPy等工具进行自动提示优化。

## 项目概述

本项目实现了论文《Large Language Models are Zero-Shot Reasoners》和《Self-Consistency Improves Chain-of-Thought Reasoning》中提出的关键技术，包括：

- **多种提示策略对比**：零样本提示、少样本提示、思维链提示、零样本思维链、自一致性
- **多任务支持**：算术推理、常识推理、逻辑推理
- **自动提示优化**：使用DSPy进行程序化提示优化
- **交互式演示界面**：实时比较不同策略的效果

## 功能特性

### 🔍 策略比较
- 实时对比不同提示策略在相同问题上的表现
- 可视化展示响应时间、推理质量、成本等指标
- 支持自一致性采样与多数投票

### 🚀 自动优化
- 使用DSPy自动搜索最优提示策略
- 多步骤推理和跨领域分析
- 智能提示模板生成

### 📈 批量评估
- 在多个问题上批量评估策略性能
- 生成详细的评估报告和分析见解
- 成本效益分析和性能排名

## 安装和使用

### 环境要求
- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd icl
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 设置DeepSeek API密钥：
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

4. 设置OpenAI API密钥（可选）：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 运行系统

1. 启动Web界面：
```bash
streamlit run app.py
```

2. 在浏览器中打开 `http://localhost:8501`

3. 配置任务类型和提示策略，开始测试

### 代码结构

```
icl/
├── app.py                 # Streamlit Web界面
├── config.py             # 配置文件和任务定义
├── model_inference.py    # 模型推理和提示工程
├── evaluation.py         # 评估和比较分析
├── dspy_integration.py   # DSPy集成和自动优化
├── requirements.txt      # 依赖包列表
└── README.md            # 项目文档
```

## 支持的提示策略

### 1. 零样本提示 (Zero-shot)
- 直接提问，不提供示例
- 快速但可能缺乏上下文

### 2. 少样本提示 (Few-shot)
- 提供少量示例进行上下文学习
- 需要精心选择示例

### 3. 思维链提示 (Chain-of-Thought)
- 引导模型进行逐步推理
- 提高复杂问题的解决能力

### 4. 零样本思维链 (Zero-shot-CoT)
- 使用"让我们一步一步思考"触发推理
- 无需示例即可获得推理过程

### 5. 自一致性 (Self-Consistency)
- 生成多个推理路径并选择最一致的答案
- 提高准确率但增加计算成本

## 任务类型

### 算术推理
- 多步骤数学问题
- 需要逻辑计算和推理

### 常识推理
- 基于日常知识的推理
- 需要背景常识理解

### 逻辑推理
- 基于逻辑规则的推理
- 需要形式逻辑应用

## 评估指标

- **准确率**：答案正确性
- **响应时间**：模型推理时间
- **成本**：API调用费用估算
- **推理质量**：推理过程的完整性和逻辑性

## 技术实现

### 模型集成
- OpenAI GPT系列模型
- 本地模型支持（Hugging Face）
- 可扩展的模型接口

### DSPy集成
- 自动提示策略优化
- 程序化提示搜索
- 多步骤推理支持

### 可视化分析
- 交互式图表展示
- 性能雷达图
- 成本效益分析

## 使用示例

### 单问题测试
1. 在"策略比较"标签页输入问题
2. 选择要比较的策略
3. 查看各策略的响应和性能指标

### 批量评估
1. 在"批量评估"标签页选择任务类型
2. 运行评估查看综合报告
3. 分析各策略在不同指标上的表现

### 自动优化
1. 在"自动优化"标签页输入复杂问题
2. 使用DSPy优化提示策略
3. 尝试多步骤和跨领域推理

## 扩展开发

### 添加新任务
在 `config.py` 的 `TASKS` 字典中添加新任务配置：

```python
"new_task": TaskConfig(
    name="新任务",
    description="任务描述",
    examples=[...],
    evaluation_metrics=[...]
)
```

### 添加新策略
在 `config.py` 的 `PROMPT_STRATEGIES` 字典中添加新策略：

```python
"new_strategy": PromptStrategy(
    name="新策略",
    description="策略描述",
    template="提示模板",
    parameters={...}
)
```

### 集成新模型
在 `model_inference.py` 的 `ModelInference` 类中添加新模型支持。

## 参考文献

1. Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." (2022)
2. Kojima, T., et al. "Large Language Models are Zero-Shot Reasoners." (2022)
3. Wang, X., et al. "Self-Consistency Improves Chain-of-Thought Reasoning in Language Models." (2022)

## 许可证

本项目仅供研究和学习使用。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
