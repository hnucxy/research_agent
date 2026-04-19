import os
import pymupdf4llm

from config.logger import get_logger
from utils.exceptions import DocumentParseError

logger = get_logger()


def parse_pdf_to_markdown(
    pdf_bytes: bytes,
    output_dir: str,
    base_name: str,
    file_name: str,
    file_hash: str,
) -> str:
    """
    将 PDF 转换为 Markdown 并提取其中的图表到本地。
    """
    logger.info("开始PDF解析 | file_name=%s | file_hash=%s", file_name, file_hash)

    # 先将上传的 PDF 字节流暂存为本地文件，以便 PyMuPDF 读取
    temp_pdf_path = os.path.join(output_dir, f"temp_{base_name}.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    # 定义图片保存的子目录
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    try:
        # 使用 pymupdf4llm 进行解析，开启图片写入
        md_text = pymupdf4llm.to_markdown(
            doc=temp_pdf_path,
            write_images=True,
            image_path=image_dir,
            image_format="png",
            dpi=300
        )
        logger.info("完成PDF解析 | file_name=%s | file_hash=%s", file_name, file_hash)
        return md_text
    except Exception as e:
        logger.exception("PDF解析阶段异常 | file_name=%s | file_hash=%s", file_name, file_hash)
        # 抛出自定义文档解析异常
        raise DocumentParseError(f"PDF 转换为 Markdown 失败: {str(e)}")
    finally:
        # 解析完毕后清理临时的 PDF 文件
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
