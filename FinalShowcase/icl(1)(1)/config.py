import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TaskConfig:
    """任务配置类"""
    name: str
    description: str
    examples: List[Dict[str, Any]]
    evaluation_metrics: List[str]

@dataclass
class PromptStrategy:
    """提示策略配置类"""
    name: str
    description: str
    template: str
    parameters: Dict[str, Any]

# 任务配置 - 专注于自动提示优化
TASKS = {
    "text_classification": TaskConfig(
        name="文本分类",
        description="对文本进行分类的任务",
        examples=[
            {
                "question": "这部电影的评论是正面的还是负面的？评论：'这部电影的剧情非常精彩，演员表演出色，强烈推荐！'",
                "answer": "正面",
                "reasoning": "评论中包含'精彩'、'出色'、'强烈推荐'等积极词汇，表明是正面评价"
            },
            {
                "question": "这个产品评论的情感倾向是什么？评论：'产品质量很差，使用一周就坏了，非常失望'",
                "answer": "负面",
                "reasoning": "评论中包含'很差'、'坏了'、'失望'等消极词汇，表明是负面评价"
            }
        ],
        evaluation_metrics=["accuracy", "response_time", "cost"]
    ),
    "information_extraction": TaskConfig(
        name="信息抽取",
        description="从文本中抽取特定信息",
        examples=[
            {
                "question": "从以下文本中提取人名、地点和时间：'张三将于明天在北京参加会议'",
                "answer": "人名：张三，地点：北京，时间：明天",
                "reasoning": "识别文本中的实体：'张三'是人名，'北京'是地点，'明天'是时间"
            },
            {
                "question": "提取以下文本中的产品名称和价格：'新款iPhone售价5999元，MacBook售价8999元'",
                "answer": "产品：iPhone，价格：5999元；产品：MacBook，价格：8999元",
                "reasoning": "识别产品名称'iPhone'和'MacBook'，以及对应的价格'5999元'和'8999元'"
            }
        ],
        evaluation_metrics=["accuracy", "response_time", "cost"]
    ),
    "question_answering": TaskConfig(
        name="问答任务",
        description="基于给定文本回答问题",
        examples=[
            {
                "question": "根据以下文本回答问题：'苹果公司于1976年4月1日由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩创立。' 问题：苹果公司是哪一年创立的？",
                "answer": "1976年",
                "reasoning": "文本中明确提到'1976年4月1日'，所以答案是1976年"
            },
            {
                "question": "根据以下文本回答问题：'中国的首都是北京，上海是最大的城市。' 问题：中国的首都是哪里？",
                "answer": "北京",
                "reasoning": "文本中明确说明'中国的首都是北京'"
            }
        ],
        evaluation_metrics=["accuracy", "response_time", "cost"]
    )
}

# 提示策略配置
PROMPT_STRATEGIES = {
    "zero_shot": PromptStrategy(
        name="零样本提示",
        description="直接提问，不提供示例",
        template="问题：{question}\n回答：",
        parameters={"temperature": 0.7, "max_tokens": 150}
    ),
    "few_shot": PromptStrategy(
        name="少样本提示",
        description="提供少量示例进行上下文学习",
        template="""以下是几个示例：

示例1：
问题：{example1_question}
回答：{example1_answer}

示例2：
问题：{example2_question}
回答：{example2_answer}

现在回答这个问题：
问题：{question}
回答：""",
        parameters={"temperature": 0.7, "max_tokens": 200}
    ),
    "chain_of_thought": PromptStrategy(
        name="思维链提示",
        description="引导模型进行逐步推理",
        template="""让我们一步一步地思考：

问题：{question}

首先，我们需要理解问题：{step1_guidance}
然后，分析关键信息：{step2_guidance}
接着，进行推理：{step3_guidance}
最后，得出结论：{step4_guidance}

所以答案是：""",
        parameters={"temperature": 0.5, "max_tokens": 300}
    ),
    "zero_shot_cot": PromptStrategy(
        name="零样本思维链",
        description="使用'让我们一步一步思考'触发推理",
        template="问题：{question}\n让我们一步一步地思考：",
        parameters={"temperature": 0.5, "max_tokens": 250}
    ),
    "self_consistency": PromptStrategy(
        name="自一致性",
        description="生成多个推理路径并选择最一致的答案",
        template="问题：{question}\n让我们从不同角度思考这个问题：",
        parameters={"temperature": 0.8, "max_tokens": 400, "num_samples": 5}
    )
}

# 模型配置
MODEL_CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-3.5-turbo",
        "max_tokens": 500
    },
    "deepseek": {
        "api_key": "sk-7a956f607a2c4704b8be34ca3487122a",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 500
    },
    "local": {
        "model_name": "microsoft/DialoGPT-medium",
        "device": "cpu"
    }
}

# 评估配置
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "response_time", "cost", "reasoning_quality"],
    "sample_size": 10,
    "max_retries": 3
}

# 成本配置（DeepSeek API定价）
COST_CONFIG = {
    "deepseek": {
        "input_cost_per_1k": 0.00014,  # 输入token成本/千token
        "output_cost_per_1k": 0.00028   # 输出token成本/千token
    },
    "openai": {
        "input_cost_per_1k": 0.0015,   # GPT-3.5-turbo输入成本
        "output_cost_per_1k": 0.0020    # GPT-3.5-turbo输出成本
    }
}
