from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import logging
import openai
import requests
import json
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    # 加载模型
    logger.info("加载BERT模型...")
    MODEL_NAME = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    logger.info("BERT模型加载完成")
except Exception as e:
    logger.error(f"加载BERT模型失败: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def get_text_embedding(text):
    try:
        # 对文本进行编码
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 使用[CLS]标记的输出作为文本的嵌入表示
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]
    except Exception as e:
        logger.error(f"获取文本嵌入失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def semantic_similarity(text1, text2):
    try:
        # 获取两段文本的嵌入
        embedding1 = get_text_embedding(text1)
        embedding2 = get_text_embedding(text2)
        
        # 计算余弦相似度
        similarity = 1 - cosine(embedding1, embedding2)
        return float(similarity)
    except Exception as e:
        logger.error(f"计算语义相似度失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def keyword_analysis(text, keywords):
    try:
        text_lower = text.lower()
        matches = []
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if keyword in text_lower:
                matches.append(keyword)
        return matches
    except Exception as e:
        logger.error(f"关键词分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class AutoGradingService:
    def __init__(self):
        self.api_base = "http://127.0.0.1:1234/v1"
        
    def generate_feedback(self, standard_answer, student_answer):
        try:
            logger.info("开始生成评分反馈")
            # 构建请求数据
            payload = {
                "model": "deepseek-r1:8b",
                "messages": [
                    {"role": "user", "content": "请对这个答案进行评分"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "grading_info": {
                    "standard_answer": standard_answer,
                    "student_answer": student_answer
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.info("发送评分请求到LLM API")
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300  # 增加超时时间到5分钟
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                logger.info("收到LLM API响应")
                response_data = response.json()
                
                # 解析JSON格式的反馈
                feedback_content = response_data['choices'][0]['message']['content']
                logger.info(f"原始反馈内容: {feedback_content}")
                
                try:
                    # 尝试解析JSON
                    if not feedback_content.strip().startswith('{'):
                        logger.warning("反馈内容不是有效的JSON格式，尝试提取JSON部分")
                        # 尝试提取JSON部分
                        import re
                        json_match = re.search(r'({.*})', feedback_content, re.DOTALL)
                        if json_match:
                            feedback_content = json_match.group(1)
                            logger.info(f"提取的JSON: {feedback_content}")
                    
                    feedback_json = json.loads(feedback_content)
                    logger.info("JSON解析成功")
                    
                    # 确保score是整数
                    try:
                        score = int(feedback_json['score'])
                    except (ValueError, TypeError):
                        # 如果score不是整数，尝试转换
                        try:
                            score = int(float(feedback_json['score']))
                        except (ValueError, TypeError):
                            # 如果仍然失败，使用默认值
                            logger.warning(f"无法将分数转换为整数: {feedback_json['score']}")
                            score = 70
                    
                    # 构建格式化的反馈
                    formatted_feedback = f"""
评分：{score}分

详细评价：
{feedback_json['comment']}
"""
                    return formatted_feedback, score
                except json.JSONDecodeError as json_error:
                    # 如果不是JSON格式，直接返回原始内容
                    logger.warning(f"JSON解析失败: {str(json_error)}")
                    return f"评分结果：\n\n{feedback_content}", 70
                
            except requests.exceptions.RequestException as e:
                # 如果API请求失败，返回一个基本的评分
                logger.error(f"API请求失败: {str(e)}")
                error_feedback = f"""
评分：70分

详细评价：
由于LLM服务暂时不可用，系统生成了一个基本评分。
错误信息: {str(e)}
建议稍后重试或联系管理员检查LLM服务状态。
"""
                return error_feedback, 70
            
        except Exception as e:
            logger.error(f"生成评分反馈失败: {str(e)}")
            logger.error(traceback.format_exc())
            error_feedback = f"""
评分：60分

详细评价：
系统在处理您的请求时遇到了问题: {str(e)}
这是一个自动生成的基本评分，建议稍后重试。
"""
            return error_feedback, 60

@app.route('/api/auto-grade', methods=['POST'])
def auto_grade():
    try:
        logger.info("收到自动评分请求")
        data = request.json
        
        student_answer = data.get('studentAnswer', '')
        standard_answer = data.get('standardAnswer', '')
        keywords = data.get('keywords', [])
        
        if not student_answer or not standard_answer:
            logger.warning("答案为空")
            return jsonify({
                'success': False,
                'message': '答案不能为空'
            }), 400
        
        # 计算语义相似度
        logger.info("计算语义相似度")
        similarity = semantic_similarity(student_answer, standard_answer)
        similarity_score = similarity * 100
        
        # 关键词匹配
        logger.info("进行关键词匹配")
        matched_keywords = keyword_analysis(student_answer, keywords)
        keyword_score = len(matched_keywords) / len(keywords) * 100 if keywords else 0
        
        # 使用本地LLM生成详细评语
        logger.info("使用LLM生成详细评语")
        grading_service = AutoGradingService()
        llm_feedback, llm_score = grading_service.generate_feedback(standard_answer, student_answer)
        
        # 确保llm_score是整数
        try:
            llm_score = int(llm_score)
        except (ValueError, TypeError):
            logger.warning(f"LLM分数不是整数: {llm_score}，使用默认值70")
            llm_score = 70
        
        # 计算最终得分（50% LLM评分 + 30%语义相似度 + 20%关键词匹配）
        final_score = (llm_score * 0.5) + (similarity_score * 0.3) + (keyword_score * 0.2)
        logger.info(f"计算最终得分: {final_score}")
        
        # 生成基础评分信息
        basic_feedback = f"自动评分结果：\n"
        basic_feedback += f"语义相似度：{similarity_score:.2f}%\n"
        basic_feedback += f"关键词匹配：{len(matched_keywords)}/{len(keywords)}\n"
        basic_feedback += f"匹配到的关键词：{', '.join(matched_keywords)}\n"
        
        if similarity_score < 60:
            basic_feedback += "\n警告：答案相似度较低，建议人工复查。"
        
        # 组合两种评语
        combined_feedback = f"{basic_feedback}\n\n{llm_feedback}"
        
        result = {
            'success': True,
            'score': round(final_score),
            'similarity': similarity_score,
            'matchedKeywords': matched_keywords,
            'feedback': combined_feedback
        }
        
        logger.info("评分完成，返回结果")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"自动评分失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'评分失败: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    logger.info("启动自动评分服务...")
    app.run(port=5000, debug=True) 