import time
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from model_inference import ModelInference, PromptEngine, SelfConsistencyEngine
from config import TASKS, PROMPT_STRATEGIES, EVALUATION_CONFIG

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    strategy_name: str
    task_type: str
    accuracy: float
    avg_response_time: float
    avg_cost: float
    reasoning_quality: float
    total_questions: int
    correct_answers: int
    details: List[Dict[str, Any]]

class Evaluator:
    """评估器类"""
    
    def __init__(self, model_inference: ModelInference):
        self.model_inference = model_inference
        self.prompt_engine = PromptEngine()
        self.self_consistency_engine = SelfConsistencyEngine(model_inference)
    
    def evaluate_strategy(self, strategy_name: str, task_type: str, 
                         questions: List[Dict[str, Any]]) -> EvaluationResult:
        """评估特定策略在任务上的表现"""
        results = []
        correct_count = 0
        total_response_time = 0
        total_cost = 0
        total_reasoning_quality = 0
        
        for i, question_data in enumerate(questions):
            question = question_data["question"]
            expected_answer = question_data["answer"]
            
            # 生成提示
            prompt = self.prompt_engine.format_prompt(strategy_name, question, task_type)
            
            # 特殊处理自一致性策略
            if strategy_name == "self_consistency":
                num_samples = PROMPT_STRATEGIES[strategy_name].parameters.get("num_samples", 5)
                actual_answer, all_responses = self.self_consistency_engine.generate_consistent_answer(
                    prompt, num_samples
                )
                response_time = 0  # 简化处理
                cost = 0  # 简化处理
            else:
                # 普通策略
                actual_answer, response_time, cost = self.model_inference.generate_response(
                    prompt, PROMPT_STRATEGIES[strategy_name].parameters
                )
                all_responses = [actual_answer]
            
            # 评估答案正确性
            is_correct = self._evaluate_answer(actual_answer, expected_answer, task_type)
            if is_correct:
                correct_count += 1
            
            # 评估推理质量
            reasoning_quality = self._evaluate_reasoning_quality(actual_answer, question, task_type)
            
            total_response_time += response_time
            total_cost += cost
            total_reasoning_quality += reasoning_quality
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "is_correct": is_correct,
                "response_time": response_time,
                "cost": cost,
                "reasoning_quality": reasoning_quality,
                "prompt": prompt,
                "all_responses": all_responses
            })
            
            print(f"进度: {i+1}/{len(questions)} - 正确: {is_correct}")
        
        # 计算总体指标
        accuracy = correct_count / len(questions) if questions else 0
        avg_response_time = total_response_time / len(questions) if questions else 0
        avg_cost = total_cost / len(questions) if questions else 0
        avg_reasoning_quality = total_reasoning_quality / len(questions) if questions else 0
        
        return EvaluationResult(
            strategy_name=strategy_name,
            task_type=task_type,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            avg_cost=avg_cost,
            reasoning_quality=avg_reasoning_quality,
            total_questions=len(questions),
            correct_answers=correct_count,
            details=results
        )
    
    def _evaluate_answer(self, actual: str, expected: str, task_type: str) -> bool:
        """评估答案正确性"""
        # 清理答案文本
        actual_clean = self._clean_answer(actual)
        expected_clean = self._clean_answer(expected)
        
        if task_type == "arithmetic_reasoning":
            # 对于算术问题，尝试提取数字
            actual_num = self._extract_number(actual_clean)
            expected_num = self._extract_number(expected_clean)
            if actual_num is not None and expected_num is not None:
                return actual_num == expected_num
        
        # 文本匹配（宽松匹配）
        return expected_clean in actual_clean or actual_clean in expected_clean
    
    def _evaluate_reasoning_quality(self, response: str, question: str, task_type: str) -> float:
        """评估推理质量（0-1分）"""
        score = 0.0
        
        # 检查是否包含推理过程
        reasoning_indicators = ["因为", "所以", "首先", "然后", "因此", "由于", "步骤", "推理"]
        has_reasoning = any(indicator in response for indicator in reasoning_indicators)
        
        if has_reasoning:
            score += 0.3
        
        # 检查是否结构化
        if "：" in response or "。" in response or "\n" in response:
            score += 0.2
        
        # 检查是否回答了问题
        if len(response) > 10:  # 非空响应
            score += 0.3
        
        # 检查是否包含关键信息
        if task_type == "arithmetic_reasoning" and any(char.isdigit() for char in response):
            score += 0.2
        elif task_type == "commonsense_reasoning" and len(response) > 20:
            score += 0.2
        elif task_type == "logical_reasoning" and any(word in response for word in ["如果", "那么", "因为", "所以"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _clean_answer(self, text: str) -> str:
        """清理答案文本"""
        if text is None:
            return ""
        return text.strip().lower().replace(" ", "").replace("。", "").replace("，", "")
    
    def _extract_number(self, text: str) -> float:
        """从文本中提取数字"""
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None

class ComparativeAnalysis:
    """比较分析类"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: EvaluationResult):
        """添加评估结果"""
        self.results.append(result)
    
    def generate_report(self) -> Dict[str, Any]:
        """生成比较分析报告"""
        if not self.results:
            return {}
        
        # 创建DataFrame用于分析
        data = []
        for result in self.results:
            data.append({
                "策略": PROMPT_STRATEGIES[result.strategy_name].name,
                "任务类型": TASKS[result.task_type].name,
                "准确率": result.accuracy,
                "平均响应时间": result.avg_response_time,
                "平均成本": result.avg_cost,
                "推理质量": result.reasoning_quality,
                "总问题数": result.total_questions,
                "正确数": result.correct_answers
            })
        
        df = pd.DataFrame(data)
        
        # 计算排名
        df["准确率排名"] = df["准确率"].rank(ascending=False)
        df["响应时间排名"] = df["平均响应时间"].rank(ascending=True)
        df["成本排名"] = df["平均成本"].rank(ascending=True)
        df["推理质量排名"] = df["推理质量"].rank(ascending=False)
        
        # 综合评分（加权平均）
        weights = {"准确率": 0.4, "推理质量": 0.3, "响应时间": 0.15, "成本": 0.15}
        df["综合评分"] = (
            df["准确率"] * weights["准确率"] +
            df["推理质量"] * weights["推理质量"] +
            (1 - df["平均响应时间"] / df["平均响应时间"].max()) * weights["响应时间"] +
            (1 - df["平均成本"] / df["平均成本"].max()) * weights["成本"]
        )
        df["综合排名"] = df["综合评分"].rank(ascending=False)
        
        # 最佳策略推荐（处理NaN值）
        best_overall = df.loc[df["综合评分"].fillna(0).idxmax()] if not df["综合评分"].isna().all() else df.iloc[0]
        best_accuracy = df.loc[df["准确率"].fillna(0).idxmax()] if not df["准确率"].isna().all() else df.iloc[0]
        fastest = df.loc[df["平均响应时间"].fillna(float('inf')).idxmin()] if not df["平均响应时间"].isna().all() else df.iloc[0]
        cheapest = df.loc[df["平均成本"].fillna(float('inf')).idxmin()] if not df["平均成本"].isna().all() else df.iloc[0]
        
        report = {
            "summary": {
                "total_evaluations": len(self.results),
                "best_overall_strategy": best_overall["策略"],
                "best_accuracy_strategy": best_accuracy["策略"],
                "fastest_strategy": fastest["策略"],
                "cheapest_strategy": cheapest["策略"]
            },
            "detailed_results": df.to_dict("records"),
            "analysis": self._generate_analysis(df)
        }
        
        return report
    
    def _generate_analysis(self, df: pd.DataFrame) -> Dict[str, str]:
        """生成分析见解"""
        analysis = {}
        
        # 准确率分析
        accuracy_stats = df["准确率"].describe()
        analysis["accuracy_analysis"] = f"准确率范围: {accuracy_stats['min']:.2f}-{accuracy_stats['max']:.2f}, 平均: {accuracy_stats['mean']:.2f}"
        
        # 推理质量分析
        reasoning_stats = df["推理质量"].describe()
        analysis["reasoning_analysis"] = f"推理质量范围: {reasoning_stats['min']:.2f}-{reasoning_stats['max']:.2f}, 平均: {reasoning_stats['mean']:.2f}"
        
        # 时间效率分析
        # 添加检查避免除以零或NaN值
        valid_accuracy = df["准确率"].dropna()
        valid_time = df["平均响应时间"].dropna()
        
        # 只有当两个序列都有有效数据且长度相同时才计算相关性
        if len(valid_accuracy) == len(valid_time) and len(valid_accuracy) > 1:
            # 进一步检查是否有非恒定值
            if valid_accuracy.std() != 0 and valid_time.std() != 0:
                time_accuracy_corr = valid_accuracy.corr(valid_time)
                analysis["time_analysis"] = f"准确率与响应时间相关性: {time_accuracy_corr:.2f}"
            else:
                analysis["time_analysis"] = "准确率与响应时间相关性: 无法计算 (数据方差为零)"
        else:
            analysis["time_analysis"] = "准确率与响应时间相关性: 无法计算 (数据不足)"
        
        # 策略效果分析
        # 处理NaN值
        valid_scores = df["综合评分"].fillna(0)
        if not valid_scores.empty and valid_scores.sum() > 0:
            best_strategy = df.loc[valid_scores.idxmax()]
            analysis["recommendation"] = f"推荐策略: {best_strategy['策略']} (综合评分: {best_strategy['综合评分']:.2f})"
        else:
            analysis["recommendation"] = "推荐策略: 数据不足，无法推荐"
        
        return analysis

# 测试函数
def test_evaluation():
    """测试评估功能"""
    inference = ModelInference()
    evaluator = Evaluator(inference)
    analysis = ComparativeAnalysis()
    
    # 测试数据
    test_questions = [
        {"question": "这部电影的评论是正面的还是负面的？评论：'这部电影的剧情非常精彩，演员表演出色，强烈推荐！'", "answer": "正面"},
        {"question": "这个产品评论的情感倾向是什么？评论：'产品质量很差，使用一周就坏了，非常失望'", "answer": "负面"},
        {"question": "请分析以下评论的情感：'这个餐厅的服务很好，食物也很美味，下次还会再来'", "answer": "正面"}
    ]
    
    # 评估不同策略
    for strategy_name in ["zero_shot", "few_shot", "zero_shot_cot"]:
        print(f"\n评估策略: {PROMPT_STRATEGIES[strategy_name].name}")
        result = evaluator.evaluate_strategy(strategy_name, "text_classification", test_questions)
        analysis.add_result(result)
        
        print(f"准确率: {result.accuracy:.2f}")
        print(f"平均响应时间: {result.avg_response_time:.2f}s")
        print(f"推理质量: {result.reasoning_quality:.2f}")
    
    # 生成报告
    report = analysis.generate_report()
    print(f"\n=== 比较分析报告 ===")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_evaluation()
