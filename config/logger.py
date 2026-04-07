import os
import logging
from datetime import datetime

class DailyFileHandler(logging.Handler):
    """
    自定义的按天输出日志处理器：
    1. 只有在真正产生日志记录（emit）时，才会创建文件，避免生成无用的空文件。
    2. 每天的日志存入独立文件，如 agent_6202-13-32.log。
    3. 不做任何自动删除，长期保留。
    """
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.current_date = None
        self.file_handler = None

    def emit(self, record):
        try:
            # 获取当天的日期字符串
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 如果日期发生变化（或者第一次运行），则切换到新的日志文件
            if self.current_date != today:
                if self.file_handler:
                    self.file_handler.close()
                
                self.current_date = today
                log_file_path = os.path.join(self.log_dir, f"agent_{today}.log")
                
                # 创建新的 FileHandler
                self.file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
                # 继承主 Handler 的格式配置
                self.file_handler.setFormatter(self.formatter)
            
            # 使用内部的 FileHandler 写入日志
            if self.file_handler:
                self.file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        # 确保程序结束时关闭文件句柄
        if self.file_handler:
            self.file_handler.close()
        super().close()


def get_logger(name="ResearchAgent"):
    """
    获取全局统一配置的 Logger 实例。
    """
    logger = logging.getLogger(name)
    
    # 如果该 logger 已经配置过，则直接返回，避免重复打印
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 1. 配置文件处理器 (使用自定义的按天延时生成处理器)
    daily_file_handler = DailyFileHandler(log_dir=log_dir)
    
    # 2. 配置控制台处理器 (保持在终端查看)
    console_handler = logging.StreamHandler()
    
    # 3. 设置日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    daily_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 挂载处理器
    logger.addHandler(daily_file_handler)
    logger.addHandler(console_handler)
    
    return logger