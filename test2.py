# 解决 Windows 符号链接警告 + 自定义缓存目录
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 基础导入
import time
import json
import random
from typing import List, Dict, Tuple
from collections import Counter

# DSPy 核心导入（适配新版 API）
import dspy
from dspy import Signature, InputField, OutputField, Predict
import datasets
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score
from tiktoken import encoding_for_model
from transformers import AutoTokenizer

# ====================== 全局配置 ======================
# Ollama 模型配置
OLLAMA_MODEL_NAME = "llama3.1:8b"
OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_API_KEY = ""
OLLAMA_TEMPERATURE = 0.0  # 显式定义温度，避免访问 LM 属性

# 任务与评估配置
TASKS = ["sentiment_classification", "named_entity_recognition", "math_reasoning"]
FEW_SHOT_K = 3
SELF_CONSISTENCY_SAMPLES = 3  # 减少采样次数，加快测试
TEST_SAMPLE_SIZE = 10  # 减少样本数，快速验证

# Token 统计配置
try:
    TOKENIZER = encoding_for_model("gpt-3.5-turbo")
except:
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)


# ====================== 模型初始化（修复 temperature 问题） ======================
def init_dspy_model():
    """初始化 DSPy + Ollama 模型（显式保存温度参数）"""
    try:
        # 初始化 Ollama LM（适配新版 DSPy）
        lm = dspy.LM(
            model=f"ollama_chat/{OLLAMA_MODEL_NAME}",
            api_base=OLLAMA_API_BASE,
            api_key=OLLAMA_API_KEY,
            max_tokens=1024,
            temperature=OLLAMA_TEMPERATURE,  # 传入温度参数
            timeout=300,
        )
        # 绑定模型到 DSPy
        dspy.settings.configure(lm=lm)
        print(f"✅ Ollama模型初始化完成！模型：{OLLAMA_MODEL_NAME}，温度：{OLLAMA_TEMPERATURE}")
        return lm
    except Exception as e:
        print(f"❌ 模型初始化失败：{str(e)}")
        exit(1)


# ====================== 数据集加载（修复 CONLL03 加载问题） ======================
def load_and_process_datasets() -> Dict[str, Dataset]:
    """加载并处理数据集（修复 CONLL03 features 访问错误）"""
    datasets_dict = {}
    print("\n📥 正在加载数据集...")

    # 1. 情感分类（SST-2）
    try:
        sst2 = load_dataset(
            "glue", "sst2",
            trust_remote_code=True
        )["validation"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))
        datasets_dict["sentiment_classification"] = sst2.map(
            lambda x: {"input": x["sentence"], "output": str(x["label"])}
        ).remove_columns(["sentence", "label", "idx"])
        print("✅ SST-2 数据集加载完成")
    except Exception as e:
        print(f"❌ SST-2 加载失败：{str(e)}")

    # 2. 实体抽取（CONLL03）- 核心修复：先取 split 再访问 features
    try:
        # 加载完整 DatasetDict，再取 validation split
        conll03_dataset = load_dataset("conll2003", trust_remote_code=True)
        conll03 = conll03_dataset["validation"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

        # 正确获取标签映射（从 split 的 features 中）
        tag_map = conll03.features["ner_tags"].int2str

        def process_ner(example):
            tokens = example["tokens"]
            tags = example["ner_tags"]
            entities = []
            current_entity = None
            for token, tag in zip(tokens, tags):
                tag_name = tag_map(tag)  # 使用 split 的 features 映射标签
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

        datasets_dict["named_entity_recognition"] = conll03.map(process_ner).remove_columns(
            ["tokens", "pos_tags", "chunk_tags", "ner_tags", "idx"])
        print("✅ CONLL03 数据集加载完成")
    except Exception as e:
        print(f"❌ CONLL03 加载失败：{str(e)}")

    # 3. 数学推理（GSM8K）
    try:
        gsm8k = load_dataset(
            "gsm8k", "main",
            trust_remote_code=True
        )["test"].shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))
        datasets_dict["math_reasoning"] = gsm8k.map(
            lambda x: {"input": x["question"], "output": x["answer"].split("\n")[-1].replace("#### ", "")}
        ).remove_columns(["question", "answer"])
        print("✅ GSM8K 数据集加载完成")
    except Exception as e:
        print(f"❌ GSM8K 加载失败：{str(e)}")

    return datasets_dict


def get_few_shot_examples(dataset: Dataset, k: int) -> List[dspy.Example]:
    """获取 Few-shot 样例"""
    return [dspy.Example(**item) for item in dataset.select(range(k))]


# ====================== 提示策略模块（修复 Predict signature 问题） ======================
# 定义通用 Signature（适配新版 DSPy 的 Predict 要求）
# ---------------------- 1. 情感分类模块 ----------------------
class SentimentSignature(Signature):
    """情感分类的输入输出定义"""
    input = InputField(desc="需要判断情感的句子")
    label = OutputField(desc="情感标签，仅 0（负面）或 1（正面）")


class SentimentZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        # 用 Signature 初始化 Predict（修复缺少 signature 问题）
        self.predict = Predict(SentimentSignature)

    def forward(self, input: str) -> dspy.Prediction:
        prompt = f"""判断以下句子的情感倾向，仅输出 0（负面）或 1（正面），无需额外解释：
句子：{input}
情感标签："""
        # 调用 predict 时传入输入参数
        return self.predict(input=input, prompt=prompt)


class SentimentFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(SentimentSignature)

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"句子：{ex.input}\n情感标签：{ex.output}" for ex in self.few_shot_examples])
        prompt = f"""根据以下样例，判断新句子的情感倾向，仅输出 0（负面）或 1（正面），无需额外解释：
{examples_str}
新句子：{input}
情感标签："""
        return self.predict(input=input, prompt=prompt)


# ---------------------- 2. 实体抽取模块 ----------------------
class NERSignature(Signature):
    """实体抽取的输入输出定义"""
    input = InputField(desc="需要抽取实体的文本")
    entities = OutputField(desc="JSON格式的实体列表，key: type（PER/ORG/LOC/MISC）、text")


class NERZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(NERSignature)

    def forward(self, input: str) -> dspy.Prediction:
        prompt = f"""从以下文本中抽取人名（PER）、机构（ORG）、地点（LOC）、其他（MISC）类型的实体，输出 JSON 格式（key: type, text），无实体则输出空列表：
文本：{input}
实体 JSON："""
        return self.predict(input=input, prompt=prompt)


class NERFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(NERSignature)

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"文本：{ex.input}\n实体 JSON：{ex.output}" for ex in self.few_shot_examples])
        prompt = f"""根据以下样例，从新文本中抽取人名（PER）、机构（ORG）、地点（LOC）、其他（MISC）类型的实体，输出 JSON 格式（key: type, text），无实体则输出空列表：
{examples_str}
新文本：{input}
实体 JSON："""
        return self.predict(input=input, prompt=prompt)


# ---------------------- 3. 数学推理模块 ----------------------
class MathZeroShotSignature(Signature):
    """数学推理（零样本）的输入输出定义"""
    input = InputField(desc="数学问题")
    answer = OutputField(desc="最终答案，仅数字")


class MathCoTSignature(Signature):
    """数学推理（CoT）的输入输出定义"""
    input = InputField(desc="数学问题")
    reasoning = OutputField(desc="分步推理过程")
    answer = OutputField(desc="最终答案，用 #### 标记")


class MathZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathZeroShotSignature)

    def forward(self, input: str) -> dspy.Prediction:
        prompt = f"""解决以下数学问题，仅输出最终答案（数字），无需额外解释：
问题：{input}
答案："""
        return self.predict(input=input, prompt=prompt)


class MathFewShot(dspy.Module):
    def __init__(self, few_shot_examples: List[dspy.Example]):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.predict = Predict(MathZeroShotSignature)

    def forward(self, input: str) -> dspy.Prediction:
        examples_str = "\n".join([f"问题：{ex.input}\n答案：{ex.output}" for ex in self.few_shot_examples])
        prompt = f"""根据以下样例，解决新数学问题，仅输出最终答案（数字），无需额外解释：
{examples_str}
新问题：{input}
答案："""
        return self.predict(input=input, prompt=prompt)


class MathCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(MathCoTSignature)

    def forward(self, input: str) -> dspy.Prediction:
        prompt = f"""解决以下数学问题，请先分步写出推理过程，最后用 #### 标注最终答案（仅数字）：
问题：{input}
推理过程："""
        response = self.predict(input=input, prompt=prompt)
        # 提取最终答案
        answer = response.answer.split("####")[-1].strip() if "####" in str(response.answer) else str(
            response.answer).strip()
        return dspy.Prediction(reasoning=response.reasoning, answer=answer)


class MathSelfConsistency(dspy.Module):
    """修复 temperature 访问问题，用全局温度参数"""

    def __init__(self, num_samples: int = SELF_CONSISTENCY_SAMPLES):
        super().__init__()
        self.num_samples = num_samples
        self.cot_module = MathCoT()
        self.base_temperature = OLLAMA_TEMPERATURE  # 使用全局定义的温度，避免访问 LM 属性

    def normalize_answer(self, ans: str) -> str:
        try:
            return str(round(float(ans.replace(",", "").strip()), 2))
        except:
            return ans.strip().lower()

    def forward(self, input: str) -> dspy.Prediction:
        samples = []
        for _ in range(self.num_samples):
            # 临时调整温度（通过 LM 的 kwargs 更新）
            dspy.settings.lm.kwargs["temperature"] = 0.7
            pred = self.cot_module(input)
            samples.append((pred.reasoning, pred.answer))

        # 恢复基准温度
        dspy.settings.lm.kwargs["temperature"] = self.base_temperature

        # 多数投票
        normalized_answers = [self.normalize_answer(ans) for _, ans in samples]
        vote_result = Counter(normalized_answers).most_common(1)[0][0]

        # 汇总推理过程
        combined_reasoning = "\n--- 采样推理过程 ---\n".join(
            [f"推理 {i + 1}：{reasoning}" for i, (reasoning, _) in enumerate(samples)])
        return dspy.Prediction(
            reasoning=combined_reasoning,
            answer=vote_result,
            all_answers=normalized_answers
        )


# ====================== 评估指标计算（修复 Token 统计） ======================
def count_tokens(text: str) -> int:
    """修复 Token 统计，处理空文本"""
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
    """修复情感分类评估，处理空预测"""
    correct = 0
    valid_count = 0
    for pred, ref in zip(preds, refs):
        if not pred:
            continue
        # 提取预测中的数字
        pred_label = [c for c in str(pred) if c in ["0", "1"]]
        pred_label = pred_label[0] if pred_label else "invalid"
        if pred_label == ref:
            correct += 1
        valid_count += 1
    accuracy = correct / valid_count if valid_count > 0 else 0.0
    return {"accuracy": round(accuracy * 100, 2)}


def evaluate_ner(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """修复实体抽取评估，处理空预测"""

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
    for pred_str, ref_str, input_text in zip(preds, refs, [ex["input"] for ex in test_data]):
        if not input_text:
            continue
        valid_count += 1
        pred_entities = parse_entities(str(pred_str))
        ref_entities = parse_entities(str(ref_str))

        tokens = input_text.split()
        pred_tags = ["O"] * len(tokens)
        ref_tags = ["O"] * len(tokens)

        # 标记实体标签
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
    return {"f1_score": round(f1, 2)}


def evaluate_math(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """修复数学推理评估，处理空预测"""
    correct = 0
    valid_count = 0
    for pred, ref in zip(preds, refs):
        if not pred or not ref:
            continue

        def normalize(ans: str) -> str:
            try:
                return str(round(float(ans.replace(",", "").strip()), 2))
            except:
                return ans.strip().lower()

        if normalize(str(pred)) == normalize(ref):
            correct += 1
        valid_count += 1
    accuracy = correct / valid_count if valid_count > 0 else 0.0
    return {"accuracy": round(accuracy * 100, 2)}


# ====================== 实验主函数（修复推理逻辑） ======================
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
            # 调用模块推理（适配新版模块返回格式）
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

            # 统计 Token（从 predict 的 prompt 中获取）
            prompt = module.predict.prompt if hasattr(module, "predict") and hasattr(module.predict, "prompt") else ""
            prompt_tokens.append(count_tokens(prompt))
            output_tokens.append(count_tokens(pred_result))

            if (i + 1) % 5 == 0:
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


# ====================== 结果展示与 Demo（保持不变） ======================
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
            print(f"实体：{pred.entities}")
        elif task_name == "math_reasoning":
            print(f"答案：{pred.answer}")
            if hasattr(pred, "reasoning"):
                print(f"\n【🧠 推理过程】\n{pred.reasoning}")

        print(f"\n【📊 性能】")
        prompt = module.predict.prompt if hasattr(module, "predict") else ""
        print(f"Prompt Token：{count_tokens(prompt)}")
        print(
            f"Output Token：{
            count_tokens(
                str(pred.answer if task_name == 'math_reasoning' else pred.label if task_name == 'sentiment_classification' else pred.entities)
            )
            }")
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
            # 选择任务支持的策略
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