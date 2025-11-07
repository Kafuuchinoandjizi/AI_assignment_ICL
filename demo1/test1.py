import dspy

 # 定义并设置大模型
model_name = 'llama3.1:8b'
lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)

# 定义输入输出参数 类定义方式
class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

question = "what is the color of the sea?"
summarize = dspy.ChainOfThought(QA)
response = summarize(question=question)

print(f"问题：{question} \n答案：{response.answer}")