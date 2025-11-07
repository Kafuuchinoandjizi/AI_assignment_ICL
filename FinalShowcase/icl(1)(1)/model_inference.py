import time
import json
import openai
from typing import Dict, List, Any, Tuple
import random
import requests
import tiktoken
from config import PROMPT_STRATEGIES, TASKS, MODEL_CONFIG, COST_CONFIG

class ModelInference:
    """模型推理类"""
    
    def __init__(self, model_type: str = "deepseek"):
        self.model_type = model_type
        self.config = MODEL_CONFIG[model_type]
        
        if model_type == "openai":
            openai.api_key = self.config["api_key"]
        elif model_type == "deepseek":
            # DeepSeek API配置
            self.deepseek_api_key = self.config["api_key"]
            self.deepseek_base_url = self.config["base_url"]
    
    def generate_response(self, prompt: str, strategy_params: Dict[str, Any]) -> Tuple[str, float, float]:
        """生成模型响应"""
        start_time = time.time()
        
        try:
            if self.model_type == "openai":
                response = openai.ChatCompletion.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=strategy_params.get("temperature", 0.7),
                    max_tokens=strategy_params.get("max_tokens", 300),  # 增加最大token数
                    timeout=60  # 增加超时时间
                )
                content = response.choices[0].message.content.strip()
                cost = self._calculate_cost(response)
            
            elif self.model_type == "deepseek":
                content, cost = self._call_deepseek_api(prompt, strategy_params)
            
            else:
                # 模拟本地模型响应（实际使用时需要替换为真实模型）
                content = self._simulate_local_response(prompt)
                cost = 0.0
            
            # 检查响应是否完整
            if self._is_response_complete(content, prompt):
                response_time = time.time() - start_time
                return content, response_time, cost
            else:
                # 如果响应不完整，重试一次
                print("检测到不完整响应，正在重试...")
                return self._retry_generate_response(prompt, strategy_params, start_time)
            
        except Exception as e:
            print(f"模型推理错误: {e}")
            return f"错误: {str(e)}", time.time() - start_time, 0.0
    
    def _is_response_complete(self, response: str, prompt: str) -> bool:
        """检查响应是否完整"""
        if not response or len(response.strip()) < 10:
            return False
        
        # 检查是否包含常见的结束标记
        if response.endswith(('.', '。', '!', '！', '?', '？')):
            return True
        
        # 检查是否回答了问题
        question_keywords = ['什么', '如何', '为什么', '多少', '哪个', '谁']
        if any(keyword in prompt for keyword in question_keywords):
            # 如果问题包含疑问词，响应应该相对完整
            return len(response) > 20
        
        return True
    
    def _retry_generate_response(self, prompt: str, strategy_params: Dict[str, Any], start_time: float) -> Tuple[str, float, float]:
        """重试生成响应"""
        try:
            # 增加最大token数并降低温度以获得更稳定的响应
            retry_params = strategy_params.copy()
            retry_params["max_tokens"] = retry_params.get("max_tokens", 150) + 100
            retry_params["temperature"] = retry_params.get("temperature", 0.7) * 0.8
            
            if self.model_type == "openai":
                response = openai.ChatCompletion.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    **retry_params
                )
                content = response.choices[0].message.content.strip()
                cost = self._calculate_cost(response)
            elif self.model_type == "deepseek":
                content, cost = self._call_deepseek_api(prompt, retry_params)
            else:
                content = self._simulate_local_response(prompt)
                cost = 0.0
            
            response_time = time.time() - start_time
            return content, response_time, cost
            
        except Exception as e:
            print(f"重试失败: {e}")
            return "响应生成失败，请重试", time.time() - start_time, 0.0
    
    def _calculate_cost(self, response) -> float:
        """计算OpenAI API调用成本"""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_config = COST_CONFIG["openai"]
        input_cost = (input_tokens / 1000) * cost_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * cost_config["output_cost_per_1k"]
        return input_cost + output_cost
    
    def _call_deepseek_api(self, prompt: str, strategy_params: Dict[str, Any]) -> Tuple[str, float]:
        """调用DeepSeek API"""
        if not self.deepseek_api_key:
            return "请设置DEEPSEEK_API_KEY环境变量", 0.0
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        
        data = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": strategy_params.get("temperature", 0.7),
            "max_tokens": strategy_params.get("max_tokens", 150),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # 计算DeepSeek成本（基于token估算）
                cost = self._estimate_deepseek_cost(prompt, content)
                return content, cost
            else:
                error_msg = f"DeepSeek API错误: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg, 0.0
                
        except requests.exceptions.RequestException as e:
            error_msg = f"DeepSeek API请求失败: {e}"
            print(error_msg)
            return error_msg, 0.0
    
    def _estimate_deepseek_cost(self, prompt: str, response: str) -> float:
        """估算DeepSeek API调用成本"""
        try:
            # 使用tiktoken估算token数量
            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(prompt))
            output_tokens = len(encoding.encode(response))
            
            cost_config = COST_CONFIG["deepseek"]
            input_cost = (input_tokens / 1000) * cost_config["input_cost_per_1k"]
            output_cost = (output_tokens / 1000) * cost_config["output_cost_per_1k"]
            
            return input_cost + output_cost
        except:
            # 如果tiktoken不可用，使用字符数估算
            input_chars = len(prompt)
            output_chars = len(response)
            # 近似估算：1 token ≈ 4个字符
            input_tokens = input_chars / 4
            output_tokens = output_chars / 4
            
            cost_config = COST_CONFIG["deepseek"]
            input_cost = (input_tokens / 1000) * cost_config["input_cost_per_1k"]
            output_cost = (output_tokens / 1000) * cost_config["output_cost_per_1k"]
            
            return input_cost + output_cost
    
    def _simulate_local_response(self, prompt: str) -> str:
        """模拟本地模型响应"""
        # 这里可以集成Hugging Face模型或其他本地模型
        responses = [
            "根据我的分析，这个问题的答案是...",
            "让我思考一下这个问题...",
            "基于提供的信息，我认为...",
            "经过推理，我得出的结论是...",
            "这个问题可以从多个角度考虑..."
        ]
        return random.choice(responses) + " [模拟响应]"

class PromptEngine:
    """提示工程引擎"""
    
    def __init__(self):
        self.strategies = PROMPT_STRATEGIES
    
    def format_prompt(self, strategy_name: str, question: str, task_type: str = None) -> str:
        """根据策略格式化提示"""
        strategy = self.strategies[strategy_name]
        template = strategy.template
        
        if strategy_name == "zero_shot":
            return template.format(question=question)
        
        elif strategy_name == "few_shot":
            if task_type and task_type in TASKS:
                examples = TASKS[task_type].examples
                if len(examples) >= 2:
                    return template.format(
                        example1_question=examples[0]["question"],
                        example1_answer=examples[0]["answer"],
                        example2_question=examples[1]["question"],
                        example2_answer=examples[1]["answer"],
                        question=question
                    )
            # 默认示例
            return template.format(
                example1_question="2 + 2等于多少？",
                example1_answer="4",
                example2_question="太阳从哪边升起？",
                example2_answer="东边",
                question=question
            )
        
        elif strategy_name == "chain_of_thought":
            # 根据问题类型提供不同的引导
            guidance = self._get_cot_guidance(question, task_type)
            return template.format(
                question=question,
                step1_guidance=guidance["step1"],
                step2_guidance=guidance["step2"],
                step3_guidance=guidance["step3"],
                step4_guidance=guidance["step4"]
            )
        
        elif strategy_name == "zero_shot_cot":
            return template.format(question=question)
        
        elif strategy_name == "self_consistency":
            return template.format(question=question)
        
        else:
            return f"问题：{question}\n回答："
    
    def _get_cot_guidance(self, question: str, task_type: str) -> Dict[str, str]:
        """获取思维链引导"""
        if task_type == "arithmetic_reasoning":
            return {
                "step1": "识别问题中的数字和运算关系",
                "step2": "确定需要执行的数学运算步骤",
                "step3": "按顺序执行计算",
                "step4": "验证结果是否合理"
            }
        elif task_type == "commonsense_reasoning":
            return {
                "step1": "理解问题的场景和背景",
                "step2": "回忆相关的常识知识",
                "step3": "应用常识进行推理",
                "step4": "检查推理是否符合现实"
            }
        elif task_type == "logical_reasoning":
            return {
                "step1": "分析前提条件和逻辑关系",
                "step2": "应用逻辑规则进行推导",
                "step3": "检查推导过程是否有效",
                "step4": "得出逻辑结论"
            }
        else:
            return {
                "step1": "理解问题的核心",
                "step2": "分析相关信息",
                "step3": "进行推理思考",
                "step4": "得出结论"
            }

class SelfConsistencyEngine:
    """自一致性引擎"""
    
    def __init__(self, model_inference: ModelInference):
        self.model_inference = model_inference
    
    def generate_consistent_answer(self, prompt: str, num_samples: int = 5) -> Tuple[str, List[str]]:
        """生成自一致性答案"""
        responses = []
        
        for i in range(num_samples):
            response, _, _ = self.model_inference.generate_response(
                prompt, {"temperature": 0.8, "max_tokens": 200}
            )
            responses.append(response)
        
        # 简单的多数投票（实际应用中需要更复杂的答案提取逻辑）
        answer_counts = {}
        for response in responses:
            # 简化：取响应中的第一个数字或关键词作为答案
            answer = self._extract_answer(response)
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        if answer_counts:
            consistent_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        else:
            consistent_answer = "无法确定一致答案"
        
        return consistent_answer, responses
    
    def _extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        # 简化实现，实际应用中需要更复杂的答案提取逻辑
        if "答案是" in response:
            parts = response.split("答案是")
            if len(parts) > 1:
                return parts[1].strip().split("。")[0]
        return response[:50]  # 返回前50个字符作为简化答案

# 测试函数
def test_inference():
    """测试推理功能"""
    inference = ModelInference()
    prompt_engine = PromptEngine()
    
    test_question = "小明有3个苹果，小红给了他2个苹果，现在小明有多少个苹果？"
    
    for strategy_name in PROMPT_STRATEGIES:
        print(f"\n=== {PROMPT_STRATEGIES[strategy_name].name} ===")
        prompt = prompt_engine.format_prompt(strategy_name, test_question, "arithmetic_reasoning")
        print(f"提示: {prompt[:100]}...")
        
        response, response_time, cost = inference.generate_response(
            prompt, PROMPT_STRATEGIES[strategy_name].parameters
        )
        print(f"响应: {response}")
        print(f"时间: {response_time:.2f}s, 成本: ${cost:.4f}")

if __name__ == "__main__":
    test_inference()
