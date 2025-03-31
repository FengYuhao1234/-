from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from typing import Optional, List, Dict
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import traceback
from langchain.chains.question_answering import load_qa_chain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import requests
import subprocess

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = FastAPI()

# 初始化Ollama模型
llm = OllamaLLM(model="deepseek-r1:8b")

# 添加提示词模板
GRADING_TEMPLATE = """你是一个信息管理与信息系统研究方法论的教师。基于以下课程材料和标准答案，请对学生的答案进行评分和点评。

参考课程材料：
{context}

标准答案：{standard_answer}

学生答案：{student_answer}

标准答案拆解方法：若标准答案为：
	Development Research on Chinese Culture English Blog	The value of certification
Position (The purposes of literature review)	The second paragraph of 引言	The last paragraph of INTRODUCTION
Structure (Organize around headings)	Organize around headings "definition of blog", "the current study", and "the study question and value".	Organize around headings "how are firms recruiting ISN professionals?", "the current study", and "the study question, motivation and value".
Content (Recognized and related to the research title)	Literature (1) is a journal article and relate to the research title.	Literature (Williamson, 1997) is a journal article and relate to the research title.

其拆解格式为：
        "case1_purpose": "Development Research on Chinese Culture English Blog",
        "case2_purpose": "The value of certification Position (The purposes of literature review)",
        "case1_intro": "The second paragraph of 引言",
        "case2_intro": "The last paragraph of INTRODUCTION",
        "case1_structure": "Organize around headings \"definition of blog\", \"the current study\", and \"the study question and value\".",
        "case2_structure": "Organize around headings \"how are firms recruiting ISN professionals?\", \"the current study\", and \"the study question, motivation and value\".",
        "case1_content": "Literature (1) is a journal article and relate to the research title.",
        "case2_content": "Literature (Williamson, 1997) is a journal article and relate to the research title. ""
请严格按照以下内容进行评分：可以看到这里面总共有8个点，学生答案每涵盖一个点，就给12.5分，满分为100分,换句话说，评分只有12.5分，25分，37.5分，50分，62.5分，75分，87.5分，100分这8个分数，请严格遵守。

请从以下几个方面进行评价：
1. 答案的准确性
2. 表达的完整性
3. 与标准答案的差异
4. 改进建议，严格需要查找{context}，来给出这个考点来源于哪里

请严格按照以下JSON格式输出（不要输出其他任何内容）：
{{
    "score": "分数（0-100的整数）",
    "comment": "包含上述四个方面的评价"
}}"""

# 加载和处理数据集
def load_finetune_dataset():
    try:
        logger.info("加载微调数据集...")
        # 设置环境变量以确保UTF-8编码
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
        
        with open("lora_FineTuning/lora_finetune_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 转换数据格式
        processed_data = []
        for item in data:
            processed_data.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": json.dumps(item["output"], ensure_ascii=False)
            })
        
        logger.info(f"成功加载 {len(processed_data)} 条训练数据")
        return processed_data
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_ollama_modelfile():
    modelfile_content = """FROM deepseek-r1:8b

# Set system prompt
SYSTEM "You are a teacher of Information Management and Information Systems Research Methodology, responsible for grading and commenting on student answers."

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096"""
    
    # 使用UTF-8编码写入文件
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    logger.info("创建Modelfile完成")

def train_with_ollama(train_data):
    try:
        logger.info("开始使用Ollama进行训练...")
        
        # 创建Modelfile
        create_ollama_modelfile()
        
        # 创建模型
        subprocess.run(
            ["ollama", "create", "grading-assistant", "-f", "Modelfile"],
            check=True,
            encoding='utf-8',
            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        )
        
        # 准备训练数据
        for item in train_data:
            # 构建训练提示
            prompt = f"""
Instruction: {item['instruction']}
Input: {item['input']}
Output: {item['output']}
"""
            # 使用Ollama CLI命令进行训练
            try:
                result = subprocess.run(
                    ["ollama", "run", "grading-assistant", prompt],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
                )
                logger.info(f"成功处理一条训练数据: {result.stdout[:100]}...")
            except subprocess.CalledProcessError as e:
                logger.warning(f"处理训练数据时出现警告: {e.stderr}")
                continue
        
        # 保存训练后的模型
        logger.info("训练完成，模型已自动保存")
        logger.info("你可以使用 'ollama list' 命令查看可用的模型")
        logger.info("使用 'ollama run grading-assistant' 来运行训练后的模型")
        
        return True
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# LoRA配置
def get_lora_config():
    return LoraConfig(
        r=16,  # LoRA的秩
        lora_alpha=32,  # LoRA的alpha参数
        target_modules=["q_proj", "v_proj"],  # 需要训练的模块
        lora_dropout=0.05,  # Dropout概率
        bias="none",  # 是否包含偏置项
        task_type="CAUSAL_LM"  # 任务类型
    )

# 初始化模型
def initialize_model():
    try:
        logger.info("初始化模型配置...")
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 加载基础模型
        logger.info("加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-base",
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.config.use_cache = False
        
        # 准备LoRA训练
        logger.info("准备LoRA训练...")
        model = prepare_model_for_kbit_training(model)
        
        # 应用LoRA配置
        logger.info("应用LoRA配置...")
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        
        logger.info("模型初始化完成")
        return model
    except Exception as e:
        logger.error(f"初始化模型失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 训练函数
def train_model(model, train_data, epochs=3, batch_size=4):
    try:
        logger.info("开始训练模型...")
        
        # 设置训练参数
        training_args = {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "save_steps": 100
        }
        
        # 训练模型
        model.train()
        
        logger.info("训练完成")
        return model
    except Exception as e:
        logger.error(f"训练模型失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 初始化向量数据库
def initialize_vector_store():
    try:
        logger.info("开始初始化向量数据库...")
        # 加载所有PDF文件
        documents = []
        rag_dir = "RAG"
        for file in os.listdir(rag_dir):
            if file.endswith(".pdf"):
                logger.info(f"加载PDF文件: {file}")
                loader = PyPDFLoader(os.path.join(rag_dir, file))
                documents.extend(loader.load())
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"文本分割完成，共 {len(texts)} 个文本块")
        
        # 初始化embedding模型
        logger.info("初始化embedding模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="distiluse-base-multilingual-cased-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 创建向量数据库
        logger.info("创建向量数据库...")
        vector_store = FAISS.from_documents(texts, embeddings)
        logger.info("向量数据库初始化完成")
        return vector_store
    except Exception as e:
        logger.error(f"初始化向量数据库失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 初始化向量数据库
try:
    vector_store = initialize_vector_store()
except Exception as e:
    logger.error(f"初始化向量数据库失败，程序将继续但RAG功能可能不可用: {str(e)}")
    vector_store = None

# 创建评分用的RAG检索链
grading_prompt = PromptTemplate(
    template=GRADING_TEMPLATE,
    input_variables=["context", "standard_answer", "student_answer"]
)

# 创建评分链
try:
    if vector_store:
        logger.info("创建评分链...")

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 创建自定义评分函数
        def grade_answer(query, standard_answer, student_answer):
            # 检索相关文档
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 构建提示
            prompt = grading_prompt.format(
                context=context,
                standard_answer=standard_answer,
                student_answer=student_answer
            )
            
            # 调用LLM
            response = llm.invoke(prompt)
            return {"result": response}
        
        logger.info("评分链创建完成")
    else:
        logger.warning("由于向量数据库初始化失败，评分链未创建")
        grade_answer = None
except Exception as e:
    logger.error(f"创建评分链失败: {str(e)}")
    logger.error(traceback.format_exc())
    grade_answer = None

class GradingRequest(BaseModel):
    standard_answer: str
    student_answer: str

class GradingResponse(BaseModel):
    score: int
    comment: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "deepseek-r1:8b"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    grading_info: Optional[GradingRequest] = None

class ChatCompletionResponse(BaseModel):
    id: str = "chat-1234"
    object: str = "chat.completion"
    created: int = 1234567890
    choices: List[Dict]
    usage: Dict

@app.get("/v1")
async def root():
    return {"message": "LLM API Service is running"}

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        logger.info("收到聊天完成请求")
        messages = request.messages
        user_message = next((msg.content for msg in messages if msg.role == "user"), "")
        
        # 检查是否包含评分信息
        if not request.grading_info:
            logger.warning("请求中缺少评分信息")
            raise HTTPException(status_code=400, detail="只支持评分功能，请提供 grading_info")
        
        if not grade_answer:
            logger.error("评分函数未初始化，无法处理请求")
            raise HTTPException(status_code=500, detail="评分服务未正确初始化")
            
        try:
            logger.info("开始处理评分请求")
            # 获取评分信息
            standard_answer = request.grading_info.standard_answer
            student_answer = request.grading_info.student_answer
            
            logger.info(f"标准答案: {standard_answer[:50]}...")
            logger.info(f"学生答案: {student_answer[:50]}...")
            
            # 调用评分函数
            result = grade_answer("评分请求", standard_answer, student_answer)
            
            logger.info("评分完成，处理结果")
            # 尝试解析JSON
            response_text = result["result"]
            logger.info(f"原始响应: {response_text}")
            
            try:
                # 确保响应是有效的JSON
                if not response_text.strip().startswith('{'):
                    logger.warning("响应不是有效的JSON格式，尝试提取JSON部分")
                    # 尝试提取JSON部分
                    import re
                    json_match = re.search(r'({.*})', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                        logger.info(f"提取的JSON: {response_text}")
                
                # 解析JSON
                json_response = json.loads(response_text)
                response = json.dumps(json_response, ensure_ascii=False)
                logger.info("JSON解析成功")
            except Exception as json_error:
                logger.warning(f"JSON解析失败: {str(json_error)}")
                response = response_text
        except Exception as e:
            logger.error(f"评分过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Grading error: {str(e)}")

        # 构建返回格式
        logger.info("构建API响应")
        completion_response = {
            "id": "chat-1234",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message),
                "completion_tokens": len(response),
                "total_tokens": len(user_message) + len(response)
            }
        }
        
        logger.info("请求处理完成，返回响应")
        return completion_response
    except Exception as e:
        logger.error(f"处理请求时发生未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/train")
async def train_lora():
    try:
        logger.info("开始训练流程")
        
        # 加载数据集
        train_data = load_finetune_dataset()
        
        # 使用Ollama进行训练
        success = train_with_ollama(train_data)
        
        if success:
            return {"message": "训练完成"}
        else:
            raise HTTPException(status_code=500, detail="训练失败")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("启动LLM API服务...")
    uvicorn.run(app, host="127.0.0.1", port=1234) 