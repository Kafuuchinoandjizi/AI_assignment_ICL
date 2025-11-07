# ICL提示策略对比系统

🧠 基于上下文学习（In-Context Learning）研究不同提示策略在分类、抽取、推理等任务上的效果与代价。

## 📋 项目简介

本项目是一个用于研究和比较不同提示策略（Prompt Strategies）在多种自然语言处理任务上表现的交互式系统。系统支持多种提示策略的对比分析，包括准确率、响应时间、成本等关键指标的评估。

### 核心功能

- **多策略对比**：比较少样本提示、零样本思维链、自一致性等策略
- **多任务支持**：文本分类、信息抽取、问答等任务类型
- **实时评估**：准确率、响应时间、成本等多维度评估
- **DSPy集成**：自动提示优化和程序优化功能
- **可视化分析**：交互式图表展示策略性能对比

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 支持的操作系统：Windows/Linux/macOS

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd icl原版
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥**
   - 在应用界面中配置OpenAI或DeepSeek API密钥
   - 或设置环境变量：
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"
     export DEEPSEEK_API_KEY="your-deepseek-api-key"
     ```

4. **启动应用**
   ```bash
   streamlit run app.py
   ```

5. **访问应用**
   打开浏览器访问 `http://localhost:8501`

## 📊 系统架构

### 核心模块

```
├── app.py                    # Streamlit Web应用
├── config.py                 # 配置管理（任务、策略、模型）
├── model_inference.py        # 模型推理引擎
├── evaluation.py            # 评估器
├── dspy_integration.py      # DSPy优化器
├── requirements.txt         # 依赖包
└── README.md               # 项目文档
```

### 主要组件

1. **ICLDemoApp** - 主应用类
   - 策略比较界面
   - DSPy优化界面
   - 可视化展示

2. **ModelInference** - 模型推理
   - 支持OpenAI、DeepSeek、本地模型
   - 温度采样和自一致性策略

3. **PromptEngine** - 提示工程
   - 多种提示策略格式化
   - 任务类型适配

4. **SelfConsistencyEngine** - 自一致性引擎
   - 温度采样生成多条推理路径
   - 多数投票选择最终答案

## 🎯 支持的提示策略

### 1. 少样本提示 (Few-Shot)
- **描述**：提供少量示例进行上下文学习
- **特点**：利用示例引导模型理解任务模式
- **适用场景**：需要明确任务格式的复杂任务

### 2. 零样本思维链 (Zero-Shot CoT)
- **描述**：使用"让我们一步一步思考"触发推理
- **特点**：无需示例，引导模型进行逐步推理
- **适用场景**：需要逻辑推理的复杂问题

### 3. 自一致性 (Self-Consistency)
- **描述**：生成多个推理路径并选择最一致的答案
- **特点**：使用温度采样增加多样性，多数投票提高准确性
- **适用场景**：需要高准确率的复杂推理任务

## 📈 评估指标

### 性能指标
- **准确率 (Accuracy)**：答案正确性评估
- **响应时间 (Response Time)**：模型推理耗时
- **成本 (Cost)**：API调用成本估算
- **推理质量 (Reasoning Quality)**：推理过程的完整性评估

### 任务类型
1. **文本分类** - 情感分析、主题分类等
2. **信息抽取** - 实体识别、关系抽取等  
3. **问答任务** - 基于上下文的问答

## 🔧 配置说明

### 模型配置
支持多种模型后端：
- **OpenAI GPT**：需要API密钥
- **DeepSeek**：需要API密钥  
- **本地模型**：模拟响应（可扩展）

### 任务配置
每个任务类型包含：
- 任务描述和示例
- 评估指标定义
- 预设测试问题

## 🛠️ 高级功能

### DSPy自动优化
- **自动提示优化**：根据任务类型优化提示模板
- **自动化程序优化**：代码分析和优化建议
- **自动提示搜索**：搜索最佳提示策略

### 自一致性策略
- **温度采样**：使用不同温度值生成多样推理路径
- **多数投票**：从多条路径中选择最一致的答案
- **置信度计算**：基于投票结果的置信度评估

## 📁 项目结构

```
icl原版/
├── app.py                    # 主应用文件
├── config.py                 # 配置管理
├── model_inference.py        # 模型推理
├── evaluation.py            # 评估逻辑
├── dspy_integration.py      # DSPy集成
├── requirements.txt         # 依赖列表
├── run_demo.py             # 演示脚本
├── ICL(上下文学习).pdf      # 项目文档
├── logs/                   # 日志目录
├── results/               # 结果保存
└── README.md             # 项目说明
```

## 🔍 使用示例

### 策略比较
1. 选择任务类型（文本分类/信息抽取/问答）
2. 选择要比较的提示策略
3. 输入测试问题或使用预设问题
4. 查看准确率、响应时间、成本对比

### DSPy优化
1. 在DSPy标签页中选择优化类型
2. 输入问题或代码
3. 查看优化结果和建议

## 🎓 研究背景

本项目基于以下研究论文：
- **Large Language Models are Zero-Shot Reasoners** (Chain-of-Thought)
- **Self-Consistency Improves Chain-of-Thought Reasoning**

## 🤝 贡献指南

欢迎贡献代码和想法！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目 Issues：提交问题报告
- 邮箱：提供项目维护者邮箱

## 🙏 致谢

感谢以下开源项目和工具的支持：
- [Streamlit](https://streamlit.io/) - Web应用框架
- [DSPy](https://github.com/stanfordnlp/dspy) - 程序化提示优化
- [OpenAI API](https://openai.com/) - 语言模型服务
- [DeepSeek API](https://www.deepseek.com/) - 语言模型服务

---

**注意**：使用API服务时请遵守相关服务条款，注意API调用成本控制。
