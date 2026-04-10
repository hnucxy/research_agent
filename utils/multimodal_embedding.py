import os
import dashscope
from typing import List
from langchain_core.embeddings import Embeddings

class DashScopeMultiModalEmbeddings(Embeddings):
    """
    自定义阿里云 DashScope 多模态 Embedding 接口。
    约定：如果字符串以 'image://' 开头，则解析为本地图片路径调用视觉 Embedding；否则作为文本处理。
    """
    def __init__(self, api_key: str, model: str = "multimodal-embedding-v1"):
        self.api_key = api_key
        dashscope.api_key = self.api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for item in texts:
            try:
                if item.startswith("image://"):
                    # 提取本地路径，DashScope 支持通过 file:// 协议读取本地文件
                    image_path = item.replace("image://", "")
                    abs_path = os.path.abspath(image_path)
                    input_data = [{"image": f"file://{abs_path}"}]
                else:
                    input_data = [{"text": item}]
                
                resp = dashscope.MultiModalEmbedding.call(
                    model=self.model,
                    input=input_data
                )
                
                if resp.status_code == 200:
                    embeddings.append(resp.output["embeddings"][0]["embedding"])
                else:
                    print(f"DashScope Embedding Error: {resp.message}")
                    # 维度视具体模型而定，此处为兜底占位避免报错退出
                    embeddings.append([0.0] * 1024) 
            except Exception as e:
                print(f"Exception during embedding: {str(e)}")
                embeddings.append([0.0] * 1024)
                
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # 用户检索时，默认输入的是文本 query
        return self.embed_documents([text])[0]