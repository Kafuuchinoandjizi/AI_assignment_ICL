# 环境配置与基础导入
import os
import sys
import time
import json
import random
import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass

import dspy
from dspy import Signature, InputField, OutputField, Predict
import datasets
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from tiktoken import encoding_for_model
from transformers import AutoTokenizer

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ====================== 环境配置 ======================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ====================== 全局配置 ======================
@dataclass
class Config:
    # Ollama配置
    OLLAMA_MODEL_NAME: str = "llama3.1:8b"
    OLLAMA_API_BASE: str = "http://localhost:11434"
    OLLAMA_API_KEY: str = ""
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_MAX_TOKENS: int = 1024
    OLLAMA_TIMEOUT: int = 300

    # 任务配置
    TASKS: List[str] = None
    FEW_SHOT_K: int = 3
    SELF_CONSISTENCY_SAMPLES: int = 3
    TEST_SAMPLE_SIZE: int = 100

    # 重试配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5

    # NER标签
    CONLL03_NER_TAGS: List[str] = None

    def __post_init__(self):
        if self.TASKS is None:
            self.TASKS = ["sentiment_classification", "named_entity_recognition", "math_reasoning"]
        if self.CONLL03_NER_TAGS is None:
            self.CONLL03_NER_TAGS = [
                "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                "B-LOC", "I-LOC", "B-MISC", "I-MISC"
            ]


config = Config()


# ====================== Tokenizer初始化 ======================
def init_tokenizer():
    """初始化tokenizer,优先使用GPT tokenizer"""
    try:
        tokenizer = encoding_for_model("gpt-3.5-turbo")
        logger.info("✅ 使用GPT-3.5 tokenizer")
        return tokenizer
    except Exception as e:
        logger.warning(f"⚠️  GPT tokenizer不可用,使用Llama tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                trust_remote_code=True
            )
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Tokenizer初始化失败: {e}")
            return None


TOKENIZER = init_tokenizer()


# ====================== 模型初始化 ======================
def init_dspy_model(config: Config) -> Optional[dspy.LM]:
    """初始化DSPy模型"""
    try:
        lm = dspy.LM(
            model=f"ollama_chat/{config.OLLAMA_MODEL_NAME}",
            api_base=config.OLLAMA_API_BASE,
            api_key=config.OLLAMA_API_KEY,
            max_tokens=config.OLLAMA_MAX_TOKENS,
            temperature=config.OLLAMA_TEMPERATURE,
            timeout=config.OLLAMA_TIMEOUT,
        )
        dspy.settings.configure(lm=lm)
        logger.info(f"✅ Ollama模型初始化完成:模型:{config.OLLAMA_MODEL_NAME},温度:{config.OLLAMA_TEMPERATURE}")
        return lm
    except Exception as e:
        logger.error(f"❌ 模型初始化失败:{str(e)}")
        return None


# ====================== 数据集加载(改进版) ======================
def load_dataset_with_retry(dataset_name: str, subset: Optional[str], split: str,
                            max_retries: int = 3) -> Optional[Dataset]:
    """带重试机制的数据集加载"""
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试加载 {dataset_name} ({subset or 'default'}) - 第{attempt + 1}次")
            if subset:
                ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
            logger.info(f"✅ {dataset_name} 加载成功")
            return ds
        except Exception as e:
            logger.warning(f"⚠️  加载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(config.RETRY_DELAY)
            else:
                logger.error(f"❌ {dataset_name} 加载最终失败")
                return None


def load_and_process_datasets(config: Config) -> Dict[str, Dataset]:
    """加载并处理所有数据集"""
    datasets_dict = {}
    logger.info("\n📥 正在加载数据集...")

    # 1. 情感分类(SST-2)
    sst2 = load_dataset_with_retry("glue", "sst2", "validation")
    if sst2:
        datasets_dict["sentiment_classification"] = sst2.shuffle(seed=42).select(
            range(min(config.TEST_SAMPLE_SIZE, len(sst2)))
        ).map(
            lambda x: {"input": x["sentence"], "output": str(x["label"])}
        ).remove_columns(["sentence", "label", "idx"])
        logger.info(f"✅ SST-2: {len(datasets_dict['sentiment_classification'])} 样本")

    # 2. 实体抽取(CONLL03)
    conll03 = load_dataset_with_retry("conll2003", None, "validation")
    if conll03:
        def process_ner(example):
            tokens = example["tokens"]
            tags = example["ner_tags"]
            entities = []
            current_entity = None

            for token, tag_idx in zip(tokens, tags):
                if tag_idx >= len(config.CONLL03_NER_TAGS):
                    continue
                tag_name = config.CONLL03_NER_TAGS[tag_idx]

                if tag_name.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = tag_name.split("-")[1]
                    current_entity = {"type": entity_type, "text": token}
                elif tag_name.startswith("I-") and current_entity:
                    current_entity["text"] += " " + token
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            return {
                "input": " ".join(tokens),
                "output": json.dumps(entities, ensure_ascii=False)
            }

        datasets_dict["named_entity_recognition"] = conll03.shuffle(seed=42).select(
            range(min(config.TEST_SAMPLE_SIZE, len(conll03)))
        ).map(process_ner).remove_columns(["tokens", "pos_tags", "chunk_tags", "ner_tags", "id"])
        logger.info(f"✅ CONLL03: {len(datasets_dict['named_entity_recognition'])} 样本")

    # 3. 数学推理(GSM8K)
    gsm8k = load_dataset_with_retry("gsm8k", "main", "test")
    if gsm8k:
        datasets_dict["math_reasoning"] = gsm8k.shuffle(seed=42).select(
            range(min(config.TEST_SAMPLE_SIZE, len(gsm8k)))
        ).map(
            lambda x: {
                "input": x["question"],
                "output": x["answer"].split("\n")[-1].replace("#### ", "").strip()
            }
        ).remove_columns(["question", "answer"])
        logger.info(f"✅ GSM8K: {len(datasets_dict['math_reasoning'])} 样本")

    if not datasets_dict:
        logger.error("\n🚨 警告:所有数据集加载失败。请检查网络连接或代理设置。")

    return datasets_dict


def get_few_shot_examples(dataset: Dataset, k: int) -> List[dspy.Example]:
    """获取few-shot样例"""
    return [dspy.Example(**item) for item in dataset.select(range(min(k, len(dataset))))]


# ====================== 答案归一化(改进版) ======================
def normalize_math_answer(ans: str) -> str:
    """
    增强的数学答案归一化
    处理:CoT提取、单位、分数、百分比、科学计数法
    """
    if not ans or not isinstance(ans, str):
        return ""

    # 1. 提取CoT格式答案
    if "####" in ans:
        ans = ans.split("####")[-1].strip()

    # 2. 去除货币符号和单位
    ans = re.sub(r'[$¥€£,]', '', ans)
    ans = re.sub(r'\b(dollars?|cents?|yuan|元|个|只|辆|people|years?|days?|hours?|minutes?)\b', '', ans,
                 flags=re.IGNORECASE)

    # 3. 处理百分比
    if '%' in ans:
        try:
            num = float(ans.replace('%', '').strip())
            return str(num / 100)
        except:
            pass

    # 4. 处理分数
    if '/' in ans and len(ans.split('/')) == 2:
        try:
            parts = ans.split('/')
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
            if denominator != 0:
                return str(round(numerator / denominator, 6))
        except:
            pass

    # 5. 处理科学计数法
    try:
        if 'e' in ans.lower():
            return str(float(ans))
    except:
        pass

    # 6. 标准化为浮点数
    ans = re.sub(r'[^\d.\-+]', '', ans).strip()
    try:
        num = float(ans)
        # 如果是整数,返回整数形式
        if num.is_integer():
            return str(int(num))
        return str(round(num, 6))
    except:
        return ans.strip().lower()


# ====================== Token统计 ======================
@lru_cache(maxsize=1024)
def count_tokens(text: str) -> int:
    """缓存的token计数"""
    if not text or TOKENIZER is None:
        return 0
    try:
        if hasattr(TOKENIZER, "encode"):
            return len(TOKENIZER.encode(text))
        else:
            return len(TOKENIZER(text)["input_ids"])
    except Exception as e:
        logger.warning(f"Token计数失败: {e}")
        return len(text.split())  # 降级为词数统计


# ====================== 提示策略模块 ======================
# [情感分类模块 - 保持不变]
class SentimentSignature(Signature):
    input = InputField(desc="需要判断情感的句子")
    label = OutputField(desc="情感标签,仅 0(负面)或 1(正面)")


class SentimentZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(SentimentSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""判断以下句子的情感倾向,仅输出 0(负面)或 1(正面),无需额外解释:
句子:{input}
情感标签:"""
        return self.predict(input=input, prompt=self.current_prompt)


class SentimentFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(SentimentSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"句子:{ex.input}\n情感标签:{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例,判断新句子的情感倾向,仅输出 0(负面)或 1(正面),无需额外解释:
{examples_str}
新句子:{input}
情感标签:"""
        return self.predict(input=input, prompt=self.current_prompt)


# [NER模块 - 保持不变]
class NERSignature(Signature):
    input = InputField(desc="需要抽取实体的文本")
    entities = OutputField(desc="JSON格式的实体列表,key: type(PER/ORG/LOC/MISC)、text")


class NERZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(NERSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""从以下文本中抽取人名(PER)、机构(ORG)、地点(LOC)、其他(MISC)类型的实体,输出 JSON 格式(key: type, text),无实体则输出空列表:
文本:{input}
实体 JSON:"""
        return self.predict(input=input, prompt=self.current_prompt)


class NERFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(NERSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"文本:{ex.input}\n实体 JSON:{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例,从新文本中抽取人名(PER)、机构(ORG)、地点(LOC)、其他(MISC)类型的实体,输出 JSON 格式(key: type, text),无实体则输出空列表:
{examples_str}
新文本:{input}
实体 JSON:"""
        return self.predict(input=input, prompt=self.current_prompt)


# [数学推理模块]
class MathZeroShotSignature(Signature):
    input = InputField(desc="数学问题")
    answer = OutputField(desc="最终答案,仅数字")


class MathCoTSignature(Signature):
    input = InputField(desc="数学问题")
    reasoning = OutputField(desc="分步推理过程")
    answer = OutputField(desc="最终答案,用 #### 标记")


class MathZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathZeroShotSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""解决以下数学问题,仅输出最终答案(数字),无需额外解释:
问题:{input}
答案:"""
        pred = self.predict(input=input, prompt=self.current_prompt)
        answer = normalize_math_answer(str(pred.answer))
        return dspy.Prediction(answer=answer, prompt=self.current_prompt)


class MathFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(MathZeroShotSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"问题:{ex.input}\n答案:{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例,解决新数学问题,仅输出最终答案(数字),无需额外解释:
{examples_str}
新问题:{input}
答案:"""
        pred = self.predict(input=input, prompt=self.current_prompt)
        answer = normalize_math_answer(str(pred.answer))
        return dspy.Prediction(answer=answer, prompt=self.current_prompt)


class MathCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathCoTSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""解决以下数学问题,请先分步写出推理过程,最后务必用 #### 标注最终答案(仅数字,不要单位):
问题:{input}
推理过程:"""
        response = self.predict(input=input, prompt=self.current_prompt)
        answer = normalize_math_answer(str(response.answer))
        return dspy.Prediction(reasoning=response.reasoning, answer=answer, prompt=self.current_prompt)


class MathSelfConsistency(dspy.Module):
    def __init__(self, num_samples: int = None):
        super().__init__()
        self.num_samples = num_samples or config.SELF_CONSISTENCY_SAMPLES
        self.cot_module = MathCoT()
        self.base_temperature = config.OLLAMA_TEMPERATURE
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        samples = []
        original_temp = dspy.settings.lm.kwargs.get("temperature", self.base_temperature)

        for i in range(self.num_samples):
            try:
                dspy.settings.lm.kwargs["temperature"] = 0.7
                pred = self.cot_module(input)
                samples.append((pred.reasoning, pred.answer))
                self.current_prompt = pred.prompt
            except Exception as e:
                logger.warning(f"Self-consistency采样{i + 1}失败: {e}")

        # 恢复温度
        dspy.settings.lm.kwargs["temperature"] = original_temp

        if not samples:
            return dspy.Prediction(reasoning="", answer="", all_answers=[], prompt=self.current_prompt)

        # 多数投票
        normalized_answers = [normalize_math_answer(ans) for _, ans in samples]
        valid_answers = [a for a in normalized_answers if a]

        if not valid_answers:
            vote_result = ""
        else:
            vote_result = Counter(valid_answers).most_common(1)[0][0]

        combined_reasoning = "\n--- 采样推理过程 ---\n".join(
            [f"推理 {i + 1}:{reasoning}\n答案:{ans}" for i, (reasoning, ans) in enumerate(samples)]
        )

        return dspy.Prediction(
            reasoning=combined_reasoning,
            answer=vote_result,
            all_answers=normalized_answers,
            prompt=self.current_prompt
        )


# ====================== 评估指标(改进版) ======================
def evaluate_sentiment(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """情感分类评估"""
    correct = 0
    valid_count = 0

    logger.info("\n🔍 情感分类预测验证(前5个样本):")
    for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
        if not pred:
            continue
        pred_label = [c for c in str(pred) if c in ["0", "1"]]
        pred_label = pred_label[0] if pred_label else "invalid"
        logger.info(f"  样本{i + 1}:pred={pred_label},ref={ref}")

        if pred_label == ref:
            correct += 1
        valid_count += 1

    accuracy = correct / valid_count if valid_count > 0 else 0.0
    logger.info(f"  总准确率:{round(accuracy * 100, 2)}%({correct}/{valid_count})")
    return {"accuracy": round(accuracy * 100, 2)}


def evaluate_ner(preds: List[str], refs: List[str], input_texts: List[str]) -> Dict[str, float]:
    """
    实体抽取评估(改进版)
    使用更精确的实体匹配算法
    """

    def parse_entities(json_str: str) -> List[Tuple[str, str]]:
        if not json_str:
            return []
        try:
            entities = json.loads(json_str)
            return [(ent["text"].strip(), ent["type"]) for ent in entities
                    if isinstance(ent, dict) and "text" in ent and "type" in ent]
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}, 原始字符串: {json_str[:100]}")
            return []

    all_pred_tags = []
    all_ref_tags = []

    logger.info(f"\n🔍 实体抽取预测验证(前3个样本):")

    for i, (pred_str, ref_str, input_text) in enumerate(zip(preds[:3], refs[:3], input_texts[:3])):
        if i < 3:
            logger.info(f"  样本{i + 1}:文本={input_text[:50]}...")
            logger.info(f"         pred_entities={parse_entities(str(pred_str))}")
            logger.info(f"         ref_entities={parse_entities(str(ref_str))}")

        if not input_text:
            continue

        pred_entities = parse_entities(str(pred_str))
        ref_entities = parse_entities(str(ref_str))

        tokens = input_text.split()
        pred_tags = ["O"] * len(tokens)
        ref_tags = ["O"] * len(tokens)

        # 改进的实体标记算法
        def mark_entities(tags, entities):
            marked_positions = set()
            for ent_text, ent_type in entities:
                ent_tokens = ent_text.split()
                # 使用滑动窗口查找实体
                for j in range(len(tokens) - len(ent_tokens) + 1):
                    # 检查是否已被标记
                    if any(pos in marked_positions for pos in range(j, j + len(ent_tokens))):
                        continue
                    # 检查token是否匹配
                    if tokens[j:j + len(ent_tokens)] == ent_tokens:
                        tags[j] = f"B-{ent_type}"
                        for k in range(1, len(ent_tokens)):
                            tags[j + k] = f"I-{ent_type}"
                        marked_positions.update(range(j, j + len(ent_tokens)))
                        break

        mark_entities(pred_tags, pred_entities)
        mark_entities(ref_tags, ref_entities)

        all_pred_tags.append(pred_tags)
        all_ref_tags.append(ref_tags)

    if not all_ref_tags:
        logger.warning("⚠️  没有有效的NER样本")
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    f1 = f1_score(all_ref_tags, all_pred_tags, average="micro") * 100
    precision = precision_score(all_ref_tags, all_pred_tags, average="micro") * 100
    recall = recall_score(all_ref_tags, all_pred_tags, average="micro") * 100

    logger.info(f"  F1={round(f1, 2)}%, Precision={round(precision, 2)}%, Recall={round(recall, 2)}%")

    return {
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2)
    }


def evaluate_math(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """数学推理评估(改进版)"""
    correct = 0
    valid_count = 0

    logger.info("\n🔍 数学推理预测验证(前5个样本):")
    for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
        norm_pred = normalize_math_answer(pred)
        norm_ref = normalize_math_answer(ref)
        is_correct = "✅" if norm_pred == norm_ref else "❌"

        if i < 5:
            logger.info(f"  样本{i + 1}:pred={pred} → norm={norm_pred};ref={ref} → norm={norm_ref};{is_correct}")

        if norm_ref:
            if norm_pred == norm_ref:
                correct += 1
            valid_count += 1

    accuracy = correct / valid_count if valid_count > 0 else 0.0
    logger.info(f"  总准确率:{round(accuracy * 100, 2)}%({correct}/{valid_count})")

    return {"accuracy": round(accuracy * 100, 2)}


# ====================== 实验主函数(改进版) ======================
def run_experiment(task_name: str, strategy: str, config: Config,
                   datasets_dict: Dict[str, Dataset]) -> Optional[Dict]:
    """运行单个实验"""
    if task_name not in datasets_dict or len(datasets_dict[task_name]) == 0:
        logger.error(f"❌ 任务 {task_name} 数据集未加载成功,跳过")
        return None

    test_data = datasets_dict[task_name]
    few_shot_examples = get_few_shot_examples(test_data, config.FEW_SHOT_K) if strategy == "few-shot" else []

    # 初始化模块
    try:
        if task_name == "sentiment_classification":
            module = SentimentZeroShot() if strategy == "zero-shot" else SentimentFewShot(few_shot_examples)
            evaluator = evaluate_sentiment
        elif task_name == "named_entity_recognition":
            module = NERZeroShot() if strategy == "zero-shot" else NERFewShot(few_shot_examples)
            evaluator = lambda p, r: evaluate_ner(p, r, test_data["input"])
        elif task_name == "math_reasoning":
            if strategy == "zero-shot":
                module = MathZeroShot()
            elif strategy == "few-shot":
                module = MathFewShot(few_shot_examples)
            elif strategy == "cot":
                module = MathCoT()
            elif strategy == "self-consistency":
                module = MathSelfConsistency()
            else:
                raise ValueError(f"不支持的策略: {strategy}")
            evaluator = evaluate_math
        else:
            raise ValueError(f"不支持的任务: {task_name}")
    except Exception as e:
        logger.error(f"❌ 模块初始化失败: {str(e)}")
        return None

    # 运行推理
    preds = []
    refs = []
    prompt_tokens = []
    output_tokens = []
    latencies = []
    errors = 0

    logger.info(f"\n🔄 推理 {task_name} - {strategy}({len(test_data)} 样本)...")

    for i, example in enumerate(test_data):
        input_text = example["input"]
        ref = example["output"]
        refs.append(ref)

        try:
            start_time = time.time()
            pred = module(input_text)
            latency = time.time() - start_time
            latencies.append(latency)

            # 提取预测结果
            if task_name == "sentiment_classification":
                pred_result = str(pred.label) if hasattr(pred, "label") else ""
            elif task_name == "named_entity_recognition":
                pred_result = str(pred.entities) if hasattr(pred, "entities") else ""
            elif task_name == "math_reasoning":
                pred_result = str(pred.answer) if hasattr(pred, "answer") else ""

            preds.append(pred_result)

            # 统计Token
            current_prompt = getattr(module, "current_prompt", "") or getattr(pred, "prompt", "")
            prompt_tokens.append(count_tokens(current_prompt))
            output_tokens.append(count_tokens(pred_result))

            if (i + 1) % 20 == 0:
                logger.info(f"  已完成 {i + 1}/{len(test_data)} 样本")

        except Exception as e:
            logger.warning(f"⚠️  第 {i + 1} 样本推理失败: {str(e)}")
            preds.append("")
            prompt_tokens.append(0)
            output_tokens.append(0)
            latencies.append(0)
            errors += 1

    # 计算指标
    try:
        metrics = evaluator(preds, refs)
    except Exception as e:
        logger.error(f"❌ 评估失败: {str(e)}")
        metrics = {"accuracy": 0.0} if task_name != "named_entity_recognition" else {"f1_score": 0.0}

    # 统计结果
    total_samples = len(test_data)
    avg_prompt = round(sum(prompt_tokens) / total_samples, 2) if total_samples > 0 else 0
    avg_output = round(sum(output_tokens) / total_samples, 2) if total_samples > 0 else 0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    total_tokens = sum(prompt_tokens) + sum(output_tokens)

    result = {
        "task": task_name,
        "strategy": strategy,
        "sample_size": len(test_data),
        "errors": errors,
        "accuracy/f1": metrics.get("accuracy", metrics.get("f1_score", 0)),
        "precision": metrics.get("precision", None),
        "recall": metrics.get("recall", None),
        "avg_prompt_tokens": avg_prompt,
        "avg_output_tokens": avg_output,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "preds": preds[:10],  # 只保存前10个预测用于分析
        "refs": refs[:10]
    }

    logger.info(f"✅ 任务完成:准确率/F1:{result['accuracy/f1']}%, 错误数:{errors}")
    return result


# ====================== 结果展示 ======================
def print_results_table(results: List[Dict]):
    """打印结果表格"""
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        logger.error("\n❌ 无有效实验结果")
        return

    sample_size = valid_results[0]['sample_size']

    print("\n" + "=" * 140)
    print(f"📊 实验结果汇总(样本数:{sample_size},模型:{config.OLLAMA_MODEL_NAME})")
    print("=" * 140)

    header = ["任务", "提示策略", "准确率/F1(%)", "精确率(%)", "召回率(%)",
              "Prompt Token", "Output Token", "时延(s)", "总Token", "错误数"]

    print(f"{header[0]:<20} {header[1]:<15} {header[2]:<13} {header[3]:<11} {header[4]:<10} "
          f"{header[5]:<13} {header[6]:<13} {header[7]:<10} {header[8]:<10} {header[9]:<8}")
    print("-" * 140)

    task_cn = {
        "sentiment_classification": "情感分类",
        "named_entity_recognition": "实体抽取",
        "math_reasoning": "数学推理"
    }

    for res in valid_results:
        precision = f"{res['precision']:.2f}" if res['precision'] is not None else "N/A"
        recall = f"{res['recall']:.2f}" if res['recall'] is not None else "N/A"

        print(f"{task_cn[res['task']]:<20} {res['strategy']:<15} {res['accuracy/f1']:<13.2f} "
              f"{precision:<11} {recall:<10} {res['avg_prompt_tokens']:<13.2f} "
              f"{res['avg_output_tokens']:<13.2f} {res['avg_latency']:<10.2f} "
              f"{res['total_tokens']:<10} {res['errors']:<8}")

    print("=" * 140)

    # 保存到文件
    try:
        with open("experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(valid_results, f, ensure_ascii=False, indent=2)
        logger.info("\n💾 结果已保存到 experiment_results.json")
    except Exception as e:
        logger.warning(f"⚠️  保存结果失败: {e}")


# ====================== 交互式Demo ======================
def interactive_demo(config: Config, datasets_dict: Dict[str, Dataset]):
    """交互式Demo"""
    print("\n" + "=" * 80)
    print(f"🎯 交互式Demo(模型:{config.OLLAMA_MODEL_NAME})")
    print("=" * 80)

    task_map = {
        "1": ("sentiment_classification", "情感分类(输入句子)"),
        "2": ("named_entity_recognition", "实体抽取(输入文本)"),
        "3": ("math_reasoning", "数学推理(输入题目)")
    }

    print("选择任务:")
    for k, (_, desc) in task_map.items():
        print(f"  {k}. {desc}")

    task_key = input("输入编号:").strip()
    if task_key not in task_map:
        print("❌ 无效编号")
        return

    task_name, task_desc = task_map[task_key]

    # 策略选择
    if task_name in ["sentiment_classification", "named_entity_recognition"]:
        strategy_map = {"1": "zero-shot", "2": "few-shot"}
    else:
        strategy_map = {"1": "zero-shot", "2": "few-shot", "3": "cot", "4": "self-consistency"}

    print(f"\n选择策略({task_desc}):")
    for k, s in strategy_map.items():
        print(f"  {k}. {s}")

    strategy_key = input("输入编号:").strip()
    if strategy_key not in strategy_map:
        print("❌ 无效编号")
        return

    strategy = strategy_map[strategy_key]

    # 输入文本
    input_text = input(f"\n输入{task_desc.split('(')[1].split(')')[0]}:").strip()
    if not input_text:
        print("❌ 输入不能为空")
        return

    # 加载few-shot样例
    if task_name not in datasets_dict:
        print(f"❌ 数据集 {task_name} 加载失败,无法进行Demo。")
        return

    few_shot_examples = get_few_shot_examples(datasets_dict[task_name],
                                              config.FEW_SHOT_K) if strategy == "few-shot" else []

    # 初始化模块
    try:
        if task_name == "sentiment_classification":
            module = SentimentZeroShot() if strategy == "zero-shot" else SentimentFewShot(few_shot_examples)
        elif task_name == "named_entity_recognition":
            module = NERZeroShot() if strategy == "zero-shot" else NERFewShot(few_shot_examples)
        elif task_name == "math_reasoning":
            if strategy == "zero-shot":
                module = MathZeroShot()
            elif strategy == "few-shot":
                module = MathFewShot(few_shot_examples)
            elif strategy == "cot":
                module = MathCoT()
            elif strategy == "self-consistency":
                module = MathSelfConsistency()
    except Exception as e:
        print(f"❌ 模块初始化失败:{str(e)}")
        return

    # 推理
    try:
        print("\n⏳ 正在推理...")
        start_time = time.time()
        pred = module(input_text)
        latency = time.time() - start_time

        print(f"\n【📋 结果】")
        if task_name == "sentiment_classification":
            label_map = {"0": "负面", "1": "正面"}
            label = str(pred.label)
            print(f"情感标签:{label} ({label_map.get(label, '未知')})")
        elif task_name == "named_entity_recognition":
            try:
                entities = json.loads(str(pred.entities))
                if entities:
                    print(f"实体:{json.dumps(entities, ensure_ascii=False, indent=2)}")
                else:
                    print("实体:未检测到实体")
            except:
                print(f"实体:{pred.entities}")
        elif task_name == "math_reasoning":
            print(f"答案:{pred.answer}")
            if hasattr(pred, "reasoning") and pred.reasoning:
                print(f"\n【🧠 推理过程】\n{pred.reasoning[:500]}...")
                if hasattr(pred, "all_answers") and pred.all_answers:
                    print(f"\n【🎲 Self-Consistency采样】")
                    print(f"所有答案:{pred.all_answers}")
                    print(f"投票结果:{pred.answer}")

        print(f"\n【📊 性能】")
        current_prompt = getattr(module, "current_prompt", "") or getattr(pred, "prompt", "")
        prompt_tokens = count_tokens(current_prompt)
        output = str(pred.answer if task_name == "math_reasoning" else
                     pred.label if task_name == "sentiment_classification" else pred.entities)
        output_tokens = count_tokens(output)

        print(f"Prompt Token:{prompt_tokens}")
        print(f"Output Token:{output_tokens}")
        print(f"总Token:{prompt_tokens + output_tokens}")
        print(f"时延:{round(latency, 2)}s")

    except Exception as e:
        print(f"❌ 推理失败:{str(e)}")
        logger.error(f"Demo推理错误: {e}", exc_info=True)


# ====================== 主程序入口 ======================
def main():
    """主程序"""
    print("=" * 50)
    print("🚀 启动提示策略对比实验(改进版)")
    print("=" * 50)

    # 初始化模型
    lm = init_dspy_model(config)
    if lm is None:
        logger.error("❌ 模型初始化失败,退出程序")
        return

    print("\n选择运行模式:")
    print("1. 批量运行所有实验")
    print("2. 交互式Demo")
    print("3. 运行指定任务和策略")
    mode = input("输入编号:").strip()

    # 加载数据集(所有模式都需要)
    datasets_dict = load_and_process_datasets(config)
    if not datasets_dict:
        logger.error("❌ 数据集加载失败,无法继续")
        return

    if mode == "1":
        # 批量实验
        results = []
        for task in config.TASKS:
            if task not in datasets_dict:
                logger.warning(f"⚠️  跳过未加载的任务: {task}")
                continue

            if task in ["sentiment_classification", "named_entity_recognition"]:
                task_strategies = ["zero-shot", "few-shot"]
            else:
                task_strategies = ["zero-shot", "few-shot", "cot", "self-consistency"]

            for strategy in task_strategies:
                print(f"\n" + "-" * 50)
                print(f"运行:任务={task},策略={strategy}")
                print("-" * 50)
                res = run_experiment(task, strategy, config, datasets_dict)
                if res:
                    results.append(res)

        print_results_table(results)

    elif mode == "2":
        # 交互式Demo
        while True:
            interactive_demo(config, datasets_dict)
            continue_flag = input("\n继续测试?(y/n):").strip().lower()
            if continue_flag != "y":
                print("👋 退出Demo")
                break

    elif mode == "3":
        # 运行指定实验
        print("\n可用任务:")
        for i, task in enumerate(config.TASKS, 1):
            print(f"  {i}. {task}")
        task_idx = int(input("选择任务编号:").strip()) - 1

        if 0 <= task_idx < len(config.TASKS):
            task = config.TASKS[task_idx]
            strategies = ["zero-shot", "few-shot"] if task != "math_reasoning" else ["zero-shot", "few-shot", "cot",
                                                                                     "self-consistency"]

            print(f"\n可用策略:")
            for i, s in enumerate(strategies, 1):
                print(f"  {i}. {s}")
            strategy_idx = int(input("选择策略编号:").strip()) - 1

            if 0 <= strategy_idx < len(strategies):
                strategy = strategies[strategy_idx]
                result = run_experiment(task, strategy, config, datasets_dict)
                if result:
                    print_results_table([result])
            else:
                print("❌ 无效策略编号")
        else:
            print("❌ 无效任务编号")
    else:
        print("❌ 无效模式")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n👋 用户中断程序")
    except Exception as e:
        logger.error(f"\n❌ 程序异常:{str(e)}", exc_info=True)