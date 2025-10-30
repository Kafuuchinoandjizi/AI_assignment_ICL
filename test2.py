# 环境配置与基础导入
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_RETRY_TIMEOUT"] = "30"  # 延长HF Hub超时时间，解决SSL错误

import time
import json
import random
import re
from typing import List, Dict, Tuple
from collections import Counter

import dspy
from dspy import Signature, InputField, OutputField, Predict
import datasets
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score
from tiktoken import encoding_for_model
from transformers import AutoTokenizer

# ====================== 全局配置 ======================
# Ollama 配置
OLLAMA_MODEL_NAME = "llama3.1:8b"
OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_API_KEY = ""
OLLAMA_TEMPERATURE = 0.0

# 任务与评估配置
TASKS = ["sentiment_classification", "named_entity_recognition", "math_reasoning"]
FEW_SHOT_K = 3
SELF_CONSISTENCY_SAMPLES = 3
TEST_SAMPLE_SIZE = 20  # 增加样本数，提升结果可信度
CONLL03_NER_TAGS = [  # 手动定义CONLL03标签映射（解决int2str问题）
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]

# Token 统计配置
try:
    TOKENIZER = encoding_for_model("gpt-3.5-turbo")
except:
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)


# ====================== 模型初始化 ======================
def init_dspy_model():
    try:
        lm = dspy.LM(
            model=f"ollama_chat/{OLLAMA_MODEL_NAME}",
            api_base=OLLAMA_API_BASE,
            api_key=OLLAMA_API_KEY,
            max_tokens=1024,
            temperature=OLLAMA_TEMPERATURE,
            timeout=300,
        )
        dspy.settings.configure(lm=lm)
        print(f"✅ Ollama模型初始化完成！模型：{OLLAMA_MODEL_NAME}，温度：{OLLAMA_TEMPERATURE}")
        return lm
    except Exception as e:
        print(f"❌ 模型初始化失败：{str(e)}")
        exit(1)


# ====================== 数据集加载（彻底修复CONLL03） ======================
def load_and_process_datasets() -> Dict[str, Dataset]:
    datasets_dict = {}
    print("\n📥 正在加载数据集...")

    # 1. 情感分类（SST-2）- 增加网络异常处理
    try:
        sst2 = load_dataset(
            "glue", "sst2",
            trust_remote_code=True,
            timeout=60  # 延长超时，解决SSL错误
        )["validation"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))
        datasets_dict["sentiment_classification"] = sst2.map(
            lambda x: {"input": x["sentence"], "output": str(x["label"])}
        ).remove_columns(["sentence", "label", "idx"])
        print("✅ SST-2 数据集加载完成")
    except Exception as e:
        print(f"❌ SST-2 加载失败：{str(e)}")
        print("💡 建议：检查网络代理，或重试加载")

    # 2. 实体抽取（CONLL03）- 手动标签映射，解决Sequence无int2str
    try:
        conll03_dataset = load_dataset(
            "conll2003",
            trust_remote_code=True,
            timeout=60
        )
        conll03 = conll03_dataset["validation"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

        def process_ner(example):
            tokens = example["tokens"]
            tags = example["ner_tags"]  # tags是0-8的整数列表
            entities = []
            current_entity = None

            for token, tag_idx in zip(tokens, tags):
                tag_name = CONLL03_NER_TAGS[tag_idx]  # 手动映射标签
                if tag_name.startswith("B-"):  # 实体开始
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = tag_name.split("-")[1]
                    current_entity = {"type": entity_type, "text": token}
                elif tag_name.startswith("I-") and current_entity:  # 实体继续
                    current_entity["text"] += " " + token
                else:  # 实体结束
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            if current_entity:
                entities.append(current_entity)
            return {
                "input": " ".join(tokens),
                "output": json.dumps(entities, ensure_ascii=False)
            }

        datasets_dict["named_entity_recognition"] = conll03.map(process_ner).remove_columns(
            ["tokens", "pos_tags", "chunk_tags", "ner_tags", "idx"])
        print("✅ CONLL03 数据集加载完成")
    except Exception as e:
        print(f"❌ CONLL03 加载失败：{str(e)}")

    # 3. 数学推理（GSM8K）
    try:
        gsm8k = load_dataset(
            "gsm8k", "main",
            trust_remote_code=True,
            timeout=60
        )["test"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))
        datasets_dict["math_reasoning"] = gsm8k.map(
            lambda x: {"input": x["question"], "output": x["answer"].split("\n")[-1].replace("#### ", "")}
        ).remove_columns(["question", "answer"])
        print("✅ GSM8K 数据集加载完成")
    except Exception as e:
        print(f"❌ GSM8K 加载失败：{str(e)}")

    return datasets_dict


def get_few_shot_examples(dataset: Dataset, k: int) -> List[dspy.Example]:
    return [dspy.Example(**item) for item in dataset.select(range(k))]


# ====================== 提示策略模块（修复Prompt保存与统计） ======================
# 所有模块新增 `current_prompt` 属性，保存当前推理用的prompt
# ---------------------- 1. 情感分类模块 ----------------------
class SentimentSignature(Signature):
    input = InputField(desc="需要判断情感的句子")
    label = OutputField(desc="情感标签，仅 0（负面）或 1（正面）")


class SentimentZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(SentimentSignature)
        self.current_prompt = ""  # 保存当前prompt，用于Token统计

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""判断以下句子的情感倾向，仅输出 0（负面）或 1（正面），无需额外解释：
句子：{input}
情感标签："""
        return self.predict(input=input, prompt=self.current_prompt)


class SentimentFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(SentimentSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"句子：{ex.input}\n情感标签：{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例，判断新句子的情感倾向，仅输出 0（负面）或 1（正面），无需额外解释：
{examples_str}
新句子：{input}
情感标签："""
        return self.predict(input=input, prompt=self.current_prompt)


# ---------------------- 2. 实体抽取模块 ----------------------
class NERSignature(Signature):
    input = InputField(desc="需要抽取实体的文本")
    entities = OutputField(desc="JSON格式的实体列表，key: type（PER/ORG/LOC/MISC）、text")


class NERZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(NERSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""从以下文本中抽取人名（PER）、机构（ORG）、地点（LOC）、其他（MISC）类型的实体，输出 JSON 格式（key: type, text），无实体则输出空列表：
文本：{input}
实体 JSON："""
        return self.predict(input=input, prompt=self.current_prompt)


class NERFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(NERSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"文本：{ex.input}\n实体 JSON：{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例，从新文本中抽取人名（PER）、机构（ORG）、地点（LOC）、其他（MISC）类型的实体，输出 JSON 格式（key: type, text），无实体则输出空列表：
{examples_str}
新文本：{input}
实体 JSON："""
        return self.predict(input=input, prompt=self.current_prompt)


# ---------------------- 3. 数学推理模块（增强答案归一化） ----------------------
class MathZeroShotSignature(Signature):
    input = InputField(desc="数学问题")
    answer = OutputField(desc="最终答案，仅数字")


class MathCoTSignature(Signature):
    input = InputField(desc="数学问题")
    reasoning = OutputField(desc="分步推理过程")
    answer = OutputField(desc="最终答案，用 #### 标记")


class MathZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathZeroShotSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""解决以下数学问题，仅输出最终答案（数字），无需额外解释：
问题：{input}
答案："""
        return self.predict(input=input, prompt=self.current_prompt)


class MathFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(MathZeroShotSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"问题：{ex.input}\n答案：{ex.output}" for ex in self.few_shot_examples])
        self.current_prompt = f"""根据以下样例，解决新数学问题，仅输出最终答案（数字），无需额外解释：
{examples_str}
新问题：{input}
答案："""
        return self.predict(input=input, prompt=self.current_prompt)


class MathCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathCoTSignature)
        self.current_prompt = ""

    def forward(self, input: str) -> dspy.Prediction:
        self.current_prompt = f"""解决以下数学问题，请先分步写出推理过程，最后用 #### 标注最终答案（仅数字，不要单位）：
问题：{input}
推理过程："""
        response = self.predict(input=input, prompt=self.current_prompt)
        # 提取答案（兼容模型未按 #### 标记的情况）
        answer = str(response.answer).strip()
        if "####" in answer:
            answer = answer.split("####")[-1].strip()
        # 进一步清理答案（去除括号、单位）
        answer = re.sub(r"[()（）a-zA-Z$￥元个只]", "", answer).strip()
        return dspy.Prediction(reasoning=response.reasoning, answer=answer, prompt=self.current_prompt)


class MathSelfConsistency(dspy.Module):
    def __init__(self, num_samples: int = SELF_CONSISTENCY_SAMPLES):
        super().__init__()
        self.num_samples = num_samples
        self.cot_module = MathCoT()
        self.base_temperature = OLLAMA_TEMPERATURE
        self.current_prompt = ""  # 保存最后一次采样的prompt

    def normalize_answer(self, ans: str) -> str:
        """增强答案归一化：处理单位、分数、科学计数法"""
        if not ans:
            return ""
        # 1. 去除无关字符（单位、括号、字母）
        ans = re.sub(r"[()（）a-zA-Z$￥元个只台辆]", "", ans).strip()
        # 2. 处理分数（如 "3/4" → 0.75）
        if "/" in ans and len(ans.split("/")) == 2:
            try:
                numerator, denominator = ans.split("/")
                return str(round(float(numerator) / float(denominator), 2))
            except:
                pass
        # 3. 处理科学计数法（如 "2e3" → 2000.0）
        try:
            return str(round(float(ans), 2))
        except:
            return ans.strip().lower()

    def forward(self, input: str) -> dspy.Prediction:
        samples = []
        for _ in range(self.num_samples):
            dspy.settings.lm.kwargs["temperature"] = 0.7
            pred = self.cot_module(input)
            samples.append((pred.reasoning, pred.answer))
            self.current_prompt = pred.prompt  # 保存prompt

        # 恢复温度
        dspy.settings.lm.kwargs["temperature"] = self.base_temperature

        # 多数投票
        normalized_answers = [self.normalize_answer(ans) for _, ans in samples]
        # 过滤空答案后投票
        valid_answers = [a for a in normalized_answers if a]
        if not valid_answers:
            vote_result = ""
        else:
            vote_result = Counter(valid_answers).most_common(1)[0][0]

        # 汇总推理
        combined_reasoning = "\n--- 采样推理过程 ---\n".join(
            [f"推理 {i + 1}：{reasoning}\n答案：{ans}" for i, (reasoning, ans) in enumerate(samples)])
        return dspy.Prediction(
            reasoning=combined_reasoning,
            answer=vote_result,
            all_answers=normalized_answers,
            prompt=self.current_prompt
        )


# ====================== 评估指标计算（修复统计与可信度验证） ======================
def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        if hasattr(TOKENIZER, "encode"):
            return len(TOKENIZER.encode(text))
        else:
            return len(TOKENIZER(text)["input_ids"])
    except:
        return 0


def evaluate_sentiment(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """增加调试信息，验证100%准确率真实性"""
    correct = 0
    valid_count = 0
    # 打印前5个pred和ref对比
    print("\n🔍 情感分类预测验证（前5个样本）：")
    for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
        print(f"  样本{i + 1}：pred={pred}，ref={ref}")
        if not pred:
            continue
        pred_label = [c for c in str(pred) if c in ["0", "1"]]
        pred_label = pred_label[0] if pred_label else "invalid"
        if pred_label == ref:
            correct += 1
        valid_count += 1
    accuracy = correct / valid_count if valid_count > 0 else 0.0
    print(f"  总准确率：{round(accuracy * 100, 2)}%（{correct}/{valid_count}）")
    return {"accuracy": round(accuracy * 100, 2)}


def evaluate_ner(preds: List[str], refs: List[str]) -> Dict[str, float]:
    def parse_entities(json_str: str) -> List[Tuple[str, str]]:
        if not json_str:
            return []
        try:
            entities = json.loads(json_str)
            return [(ent["text"], ent["type"]) for ent in entities if
                    isinstance(ent, dict) and "text" in ent and "type" in ent]
        except:
            return []

    all_pred_tags = []
    all_ref_tags = []
    global test_data
    valid_count = 0
    print("\n🔍 实体抽取预测验证（前2个样本）：")
    for i, (pred_str, ref_str, input_text) in enumerate(
            zip(preds[:2], refs[:2], [ex["input"] for ex in test_data[:2]])):
        print(f"  样本{i + 1}：文本={input_text[:50]}...")
        print(f"         pred_entities={parse_entities(str(pred_str))}")
        print(f"         ref_entities={parse_entities(str(ref_str))}")
        if not input_text:
            continue
        valid_count += 1
        pred_entities = parse_entities(str(pred_str))
        ref_entities = parse_entities(str(ref_str))

        tokens = input_text.split()
        pred_tags = ["O"] * len(tokens)
        ref_tags = ["O"] * len(tokens)

        # 标记实体
        for ent_text, ent_type in pred_entities:
            ent_tokens = ent_text.split()
            for i in range(len(tokens) - len(ent_tokens) + 1):
                if tokens[i:i + len(ent_tokens)] == ent_tokens:
                    pred_tags[i] = f"B-{ent_type}"
                    for j in range(1, len(ent_tokens)):
                        pred_tags[i + j] = f"I-{ent_type}"
        for ent_text, ent_type in ref_entities:
            ent_tokens = ent_text.split()
            for i in range(len(tokens) - len(ent_tokens) + 1):
                if tokens[i:i + len(ent_tokens)] == ent_tokens:
                    ref_tags[i] = f"B-{ent_type}"
                    for j in range(1, len(ent_tokens)):
                        ref_tags[i + j] = f"I-{ent_type}"

        all_pred_tags.append(pred_tags)
        all_ref_tags.append(ref_tags)

    f1 = f1_score(all_ref_tags, all_pred_tags, average="micro") * 100 if valid_count > 0 else 0.0
    print(f"  实体抽取F1分数：{round(f1, 2)}%")
    return {"f1_score": round(f1, 2)}


def evaluate_math(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """增加调试信息，查看答案归一化效果"""
    correct = 0
    valid_count = 0
    print("\n🔍 数学推理预测验证（前5个样本）：")
    for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
        # 归一化前后对比
        norm_pred = MathSelfConsistency().normalize_answer(pred)
        norm_ref = MathSelfConsistency().normalize_answer(ref)
        is_correct = "✅" if norm_pred == norm_ref else "❌"
        print(f"  样本{i + 1}：pred={pred} → norm={norm_pred}；ref={ref} → norm={norm_ref}；{is_correct}")
        if norm_pred and norm_ref:
            if norm_pred == norm_ref:
                correct += 1
            valid_count += 1
    accuracy = correct / valid_count if valid_count > 0 else 0.0
    print(f"  总准确率：{round(accuracy * 100, 2)}%（{correct}/{valid_count}）")
    return {"accuracy": round(accuracy * 100, 2)}


# ====================== 实验主函数（修复Prompt统计） ======================
def run_experiment(task_name: str, strategy: str) -> Dict[str, any]:
    global test_data
    datasets_dict = load_and_process_datasets()

    if task_name not in datasets_dict or len(datasets_dict[task_name]) == 0:
        print(f"❌ 任务 {task_name} 数据集未加载成功，跳过")
        return None

    test_data = datasets_dict[task_name]
    few_shot_examples = get_few_shot_examples(test_data, FEW_SHOT_K) if strategy == "few-shot" else []

    # 初始化模块
    try:
        if task_name == "sentiment_classification":
            module = SentimentZeroShot() if strategy == "zero-shot" else SentimentFewShot(few_shot_examples)
            evaluator = evaluate_sentiment
        elif task_name == "named_entity_recognition":
            module = NERZeroShot() if strategy == "zero-shot" else NERFewShot(few_shot_examples)
            evaluator = evaluate_ner
        elif task_name == "math_reasoning":
            if strategy == "zero-shot":
                module = MathZeroShot()
            elif strategy == "few-shot":
                module = MathFewShot(few_shot_examples)
            elif strategy == "cot":
                module = MathCoT()
            elif strategy == "self-consistency":
                module = MathSelfConsistency()
            evaluator = evaluate_math
        else:
            raise ValueError(f"不支持的任务：{task_name}")
    except Exception as e:
        print(f"❌ 模块初始化失败：{str(e)}")
        return None

    # 运行推理
    preds = []
    refs = []
    prompt_tokens = []
    output_tokens = []
    latencies = []
    reasoning_logs = []

    print(f"\n🔍 推理 {task_name} - {strategy}（{len(test_data)} 样本）...")
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
                reasoning_logs.append(str(pred.reasoning) if hasattr(pred, "reasoning") else "")

            preds.append(pred_result)

            # 统计Token：从模块的 current_prompt 获取（修复统计为0的问题）
            current_prompt = module.current_prompt if hasattr(module, "current_prompt") else (
                pred.prompt if hasattr(pred, "prompt") else ""
            )
            prompt_tokens.append(count_tokens(current_prompt))
            output_tokens.append(count_tokens(pred_result))

            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(test_data)} 样本")
        except Exception as e:
            print(f"❌ 第 {i + 1} 样本推理失败：{str(e)}")
            preds.append("")
            prompt_tokens.append(0)
            output_tokens.append(0)
            latencies.append(0)

    # 计算指标
    metrics = evaluator(preds, refs)
    avg_prompt = round(sum(prompt_tokens) / len(prompt_tokens), 2) if prompt_tokens else 0
    avg_output = round(sum(output_tokens) / len(output_tokens), 2) if output_tokens else 0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    total_tokens = round(sum(prompt_tokens) + sum(output_tokens), 2)

    result = {
        "task": task_name,
        "strategy": strategy,
        "sample_size": len(test_data),
        "accuracy/f1": metrics.get("accuracy", metrics.get("f1_score")),
        "avg_prompt_tokens": avg_prompt,
        "avg_output_tokens": avg_output,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "preds": preds,
        "refs": refs
    }

    print(f"✅ 任务完成！准确率/F1：{result['accuracy/f1']}%")
    return result


# ====================== 结果展示与 Demo ======================
def print_results_table(results: List[Dict[str, any]]):
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        print("\n❌ 无有效实验结果")
        return

    print("\n" + "=" * 120)
    print(f"📊 实验结果汇总（样本数：{TEST_SAMPLE_SIZE}，模型：{OLLAMA_MODEL_NAME}）")
    print("=" * 120)
    header = ["任务", "提示策略", "准确率/F1(%)", "平均Prompt Token", "平均Output Token", "平均时延(s)", "总Token数"]
    print(
        f"{header[0]:<20} {header[1]:<15} {header[2]:<15} {header[3]:<18} {header[4]:<18} {header[5]:<15} {header[6]:<10}")
    print("-" * 120)
    task_cn = {
        "sentiment_classification": "情感分类",
        "named_entity_recognition": "实体抽取",
        "math_reasoning": "数学推理"
    }
    for res in valid_results:
        print(
            f"{task_cn[res['task']]:<20} {res['strategy']:<15} {res['accuracy/f1']:<15} {res['avg_prompt_tokens']:<18} {res['avg_output_tokens']:<18} {res['avg_latency']:<15} {res['total_tokens']:<10}")
    print("=" * 120)


def interactive_demo():
    print("\n" + "=" * 80)
    print(f"🎯 交互式 Demo（模型：{OLLAMA_MODEL_NAME}）")
    print("=" * 80)

    task_map = {
        "1": ("sentiment_classification", "情感分类（输入句子）"),
        "2": ("named_entity_recognition", "实体抽取（输入文本）"),
        "3": ("math_reasoning", "数学推理（输入题目）")
    }
    print("选择任务：")
    for k, (_, desc) in task_map.items():
        print(f"  {k}. {desc}")
    task_key = input("输入编号：").strip()
    if task_key not in task_map:
        print("❌ 无效编号")
        return
    task_name, task_desc = task_map[task_key]

    # 策略选择
    if task_name in ["sentiment_classification", "named_entity_recognition"]:
        strategy_map = {"1": "zero-shot", "2": "few-shot"}
    else:
        strategy_map = {"1": "zero-shot", "2": "few-shot", "3": "cot", "4": "self-consistency"}
    print(f"\n选择策略（{task_desc}）：")
    for k, s in strategy_map.items():
        print(f"  {k}. {s}")
    strategy_key = input("输入编号：").strip()
    if strategy_key not in strategy_map:
        print("❌ 无效编号")
        return
    strategy = strategy_map[strategy_key]

    # 输入文本
    input_text = input(f"\n输入{task_desc.split('（')[1].split('）')[0]}：").strip()
    if not input_text:
        print("❌ 输入不能为空")
        return

    # 加载数据集（获取 few-shot 样例）
    datasets_dict = load_and_process_datasets()
    few_shot_examples = get_few_shot_examples(datasets_dict[task_name], FEW_SHOT_K) if strategy == "few-shot" else []

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
        print(f"❌ 模块初始化失败：{str(e)}")
        return

    # 推理
    try:
        start_time = time.time()
        pred = module(input_text)
        latency = time.time() - start_time

        print(f"\n【📋 结果】")
        if task_name == "sentiment_classification":
            print(f"情感标签：{pred.label}（0=负面，1=正面）")
        elif task_name == "named_entity_recognition":
            try:
                entities = json.loads(str(pred.entities))
                print(f"实体：{json.dumps(entities, ensure_ascii=False, indent=2)}")
            except:
                print(f"实体：{pred.entities}")
        elif task_name == "math_reasoning":
            print(f"答案：{pred.answer}")
            if hasattr(pred, "reasoning"):
                print(f"\n【🧠 推理过程】\n{pred.reasoning}")

        print(f"\n【📊 性能】")
        current_prompt = module.current_prompt if hasattr(module, "current_prompt") else (
            pred.prompt if hasattr(pred, "prompt") else ""
        )
        prompt_tokens = count_tokens(current_prompt)
        output = str(
            pred.answer if task_name == "math_reasoning" else pred.label if task_name == "sentiment_classification" else pred.entities)
        output_tokens = count_tokens(output)
        print(f"Prompt Token：{prompt_tokens}")
        print(f"Output Token：{output_tokens}")
        print(f"时延：{round(latency, 2)}s")
    except Exception as e:
        print(f"❌ 推理失败：{str(e)}")


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 启动提示策略对比实验")
    print("=" * 50)
    init_dspy_model()

    print("\n选择运行模式：")
    print("1. 批量运行所有实验")
    print("2. 交互式 Demo")
    mode = input("输入编号：").strip()

    if mode == "1":
        results = []
        for task in TASKS:
            if task in ["sentiment_classification", "named_entity_recognition"]:
                task_strategies = ["zero-shot", "few-shot"]
            else:
                task_strategies = ["zero-shot", "few-shot", "cot", "self-consistency"]

            for strategy in task_strategies:
                print(f"\n" + "-" * 50)
                print(f"运行：任务={task}，策略={strategy}")
                print("-" * 50)
                res = run_experiment(task, strategy)
                if res:
                    results.append(res)

        print_results_table(results)

    elif mode == "2":
        while True:
            interactive_demo()
            continue_flag = input("\n继续测试？（y/n）：").strip().lower()
            if continue_flag != "y":
                print("👋 退出 Demo")
                break
    else:
        print("❌ 无效模式")