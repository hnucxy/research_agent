import sys
import os

# 将项目根目录添加到系统环境变量中，确保能成功导入上一级的 config 模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import base64
from langchain_core.messages import HumanMessage
from config.settings import Settings

def encode_image(image_path):
    """将本地图片文件转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_multimodal_llm(image_path: str, prompt_text: str):
    """
    单轮对话测试视觉大模型
    """
    if not os.path.exists(image_path):
        print(f"❌ 错误: 找不到测试图片 {image_path}，请先准备一张图片。")
        return

    print(f"🔄 1. 正在读取并编码图片: {image_path}")
    base64_image = encode_image(image_path)
    
    # 动态判断图片的 mime_type
    ext = image_path.split('.')[-1].lower()
    mime_type = "image/png"
    if ext in ['jpg', 'jpeg']:
        mime_type = "image/jpeg"
    elif ext == 'webp':
        mime_type = "image/webp"

    print("🔄 2. 正在组装 LangChain 消息体...")
    # 按照 OpenAI 兼容格式组装 LangChain 的多模态 content 块
    content_blocks = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
        },
        {
            "type": "text", 
            "text": prompt_text
        }
    ]

    messages = [
        HumanMessage(content=content_blocks)
    ]

    print("🔄 3. 正在调用大语言模型 (通过 Settings.get_llm)...")
    try:
        llm = Settings.get_llm(temperature=0.1)
        response = llm.invoke(messages)
        
        print("\n" + "="*20 + " 模型返回结果 " + "="*20)
        print(response.content)
        print("="*54 + "\n")
        
    except Exception as e:
        print(f"\n❌ 模型调用失败: {e}")

if __name__ == "__main__":
    print("=== Qwen 多模态视觉模型测试 ===")
    
    # 构建测试图片的绝对路径（假设图片放在 tests 文件夹下，名为 test_image.png）
    # 你也可以根据实际情况修改这里的路径
    current_dir = os.path.dirname(__file__)
    test_image_path = os.path.join(current_dir, "test_image.png")
    
    user_prompt = "请详细描述一下这张图片中的内容，包含哪些核心要素？"
    
    if not os.path.exists(test_image_path):
        print(f"\n⚠️ 提示: 请在 '{current_dir}' 目录下放入一张名为 'test_image.png' 的图片。")
    else:
        run_multimodal_llm(test_image_path, user_prompt)
