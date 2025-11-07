#!/usr/bin/env python3
"""
ICL提示策略对比系统 - 演示启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "streamlit", "pandas", "plotly", "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """设置环境变量"""
    # 检查DeepSeek API密钥（现在已内置在配置中，此检查可选）
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ℹ️  未设置 DEEPSEEK_API_KEY 环境变量（可选，已在配置中内置）")
        print("如需覆盖默认密钥，可设置环境变量：")
        print("export DEEPSEEK_API_KEY='your-api-key-here'")
        print()
    
    # 检查OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("ℹ️  未设置 OPENAI_API_KEY 环境变量（可选）")
        print("如果需要使用OpenAI模型，请设置环境变量：")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    # 创建必要的目录
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

def run_tests():
    """运行基本测试"""
    print("🧪 运行基本测试...")
    
    try:
        # 测试配置加载
        from config import TASKS, PROMPT_STRATEGIES
        print(f"✅ 加载了 {len(TASKS)} 个任务和 {len(PROMPT_STRATEGIES)} 个策略")
        
        # 测试模型推理（模拟模式）
        from model_inference import ModelInference
        inference = ModelInference()
        print("✅ 模型推理模块初始化成功")
        
        # 测试评估模块
        from evaluation import Evaluator
        evaluator = Evaluator(inference)
        print("✅ 评估模块初始化成功")
        
        # 测试DSPy集成
        from dspy_integration import DSPyOptimizer
        optimizer = DSPyOptimizer()
        print("✅ DSPy集成模块初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def start_web_interface():
    """启动Web界面"""
    print("🚀 启动ICL提示策略对比系统...")
    print("📊 访问地址: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止服务")
    print()
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("🧠 ICL提示策略对比系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 运行测试
    if not run_tests():
        print("❌ 系统测试失败，请检查依赖和配置")
        sys.exit(1)
    
    print("✅ 所有检查通过，准备启动系统...")
    print()
    
    # 询问是否打开浏览器
    try:
        response = input("是否自动打开浏览器？(y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            print("将在Streamlit启动后自动打开浏览器...")
            # 不在这里打开浏览器，避免重复打开
            # 浏览器将在Streamlit启动后自动打开
    except KeyboardInterrupt:
        print("\n👋 用户取消")
        sys.exit(0)
    
    # 启动Web界面
    start_web_interface()

if __name__ == "__main__":
    main()
