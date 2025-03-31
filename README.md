# -
毕设项目

# 自动评分系统

这是一个基于深度学习和大语言模型的自动评分系统，专门用于信息管理与信息系统研究方法论课程的答案评分。

## 系统架构

系统由两个主要服务组成：

1. LLM API 服务 (`llm_api.py`)
   - 基于 FastAPI 框架
   - 使用 Ollama 进行本地模型推理
   - 支持 LoRA 微调训练
   - 提供评分和反馈生成功能

2. 自动评分服务 (`auto_grading_service.py`)
   - 基于 Flask 框架
   - 集成 BERT 模型进行语义相似度分析
   - 提供关键词匹配功能
   - 综合评分系统

## 功能特点

- 多维度评分：
  - LLM 评分（50%权重）
  - 语义相似度分析（30%权重）
  - 关键词匹配（20%权重）
- 自动生成详细评语
- 支持模型微调
- 实时评分反馈
- 健康检查接口

## 安装要求

1. Python 3.8+
2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 Ollama（用于本地模型推理）

## 环境变量设置

```bash
# Windows
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

## 启动服务

1. 启动 LLM API 服务：
```bash
python llm_api.py
```
服务将在 http://127.0.0.1:1234 运行

2. 启动自动评分服务：
```bash
python auto_grading_service.py
```
服务将在 http://127.0.0.1:5000 运行

## API 接口

### LLM API 服务

1. 评分接口
```
POST /v1/chat/completions
Content-Type: application/json

{
    "model": "deepseek-r1:8b",
    "messages": [
        {"role": "user", "content": "请对这个答案进行评分"}
    ],
    "grading_info": {
        "standard_answer": "标准答案",
        "student_answer": "学生答案"
    }
}
```

2. 训练接口
```
POST /v1/train
```

### 自动评分服务

1. 评分接口
```
POST /api/auto-grade
Content-Type: application/json

{
    "studentAnswer": "学生答案",
    "standardAnswer": "标准答案",
    "keywords": ["关键词1", "关键词2"]
}
```

2. 健康检查
```
GET /health
```

## 模型训练

1. 准备训练数据：
   - 将训练数据放在 `lora_FineTuning` 目录下
   - 数据格式为 JSON，包含 instruction、input 和 output

2. 启动训练：
```bash
curl -X POST http://127.0.0.1:1234/v1/train
```

## 评分规则

- 评分范围：0-100分
- LLM 评分规则：
  - 每个知识点 12.5 分
  - 总共 8 个知识点
  - 分数档位：12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100

## 注意事项

1. 确保在启动服务前已安装所有依赖
2. Windows 环境下需要正确设置编码环境变量
3. 训练数据需要使用 UTF-8 编码
4. 建议使用 GPU 进行模型训练
5. 评分结果建议进行人工复查，特别是相似度低于 60% 的答案

## 错误处理

- 服务会自动记录详细的错误日志
- 如遇到编码问题，请检查环境变量设置
- API 请求超时默认设置为 5 分钟

## 维护和支持

- 定期检查日志文件
- 监控服务健康状态
- 适时更新模型和训练数据 
