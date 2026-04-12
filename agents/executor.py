import os
import re
import base64
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from config.settings import Settings
from config.logger import get_logger
from graph.state import AgentState
from tools.academic_writer_tool import AcademicWriterTool
from tools.arxiv_tool import ArxivSearchTool
from tools.literature_reader_tool import LiteratureReaderTool
from tools.literature_rag_tool import LiteratureRagTool
from prompts.executor_prompts import TOOL_EXTRACTION_PROMPT,TEXT_GENERATION_PROMPT
from utils.exceptions import ToolExecutionError, ToolExecutionTimeout

logger = get_logger()

class ExecutorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)

        # 注册工具池（未来随意扩充）
        self.tools = {
            "arxiv_search": ArxivSearchTool(), # arxiv文献检索工具
            "academic_write": AcademicWriterTool(), # 学术内容写作工具
            "literature_read": LiteratureReaderTool(), #文献阅读工具
            "literature_rag_search": LiteratureRagTool() #RAG工具
        }

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Executor] Node ---")
        current_index = state["current_step_index"]
        current_step = state["plan"][current_index]
        tool_name = state["planned_tools"][current_index]

        #处理反馈信息
        feedback = ""
        # 检查是否是重试循环
        eval_res = state.get("evaluation_result")
        if eval_res and not  eval_res.get("passed"):
            feedback = f"\n【⚠️上一次执行未通过，评估专家给出的修正建议】: {eval_res.get('feedback')}\n请务必参考此建议调整本次的工具参数或生成逻辑！"


        # print(f"正在执行步骤: {current_step}")
        logger.info("正在执行步骤: %s", current_step)
        context_raw = "\n".join(state.get("step_history", []))
        # 预处理上下文
        context_str = context_raw if context_raw else "无"
        # output = ""

        
        # 通用路由与执行逻辑
        
        if tool_name == "memo_output":
            logger.info("    📦 识别到语义缓存，正在精准提取结论...")
            # 使用正则从冗余的记忆块中，提取最后一个 "Result:" 之后的所有内容（即真正的执行结论）
            match = re.search(r"Result:\s*(.*)", current_step, re.DOTALL)
            if match:
                output = match.group(1).strip()
            else:
                # 兜底清理：如果没匹配到 Result 标识，则尝试移除缓存标识符
                output = current_step.replace("【语义缓存直接输出】\n", "").strip()
        elif tool_name in self.tools:
            tool = self.tools[tool_name]
            # print(f"    🛠️ 准备调用外部工具: [{tool_name}]")
            logger.info("    🛠️ 准备调用外部工具: [%s]", tool_name)

            # 获取全局原始任务
            original_task = state.get("task_input", "")

            # 动态获取工具的描述。
            # 工具类有 description 属性（这是编写 Agent 工具的标准规范）
            tool_desc = getattr(tool, "description", "执行特定任务的工具")

            # [工程规范修改点]：使用 ChatPromptTemplate 组装链 (Chain)
            prompt_template = ChatPromptTemplate.from_template(TOOL_EXTRACTION_PROMPT)
            chain = prompt_template | self.llm

            # 动态传入变量字典（参数）
            query_res = chain.invoke({
                "original_task": original_task,
                "tool_name": tool_name,
                "tool_desc": tool_desc,
                "current_step": current_step,
                "context": context_str,
                "feedback": feedback
            })
            tool_input = query_res.content.strip()
            # print(f"    解析出的工具参数: [{tool_input}]")
            logger.info("    解析出的工具参数: [{%s}]", tool_input)

            # 统一执行接口
            try:
                # 未来如果加入了超时控制，可以抛出 ToolExecutionTimeout
                tool_result = tool.run(tool_input)
                output = f"【{tool_name} 执行结果】:\n{tool_result}"
            except ToolExecutionTimeout as te:
                logger.error("工具 %s 执行超时: %s", tool_name, te)
                output = f"【{tool_name} 执行超时】: 请精简参数或稍后再试。"
            except ToolExecutionError as te:
                logger.error("工具 %s 执行失败: %s", tool_name, te)
                output = f"【{tool_name} 执行失败】: {str(te)}"
            except Exception as e:
                # 兜底捕获未知错误
                logger.exception("工具 %s 发生未知的系统异常", tool_name)
                output = f"【{tool_name} 发生系统级错误】: {str(e)}"

        elif tool_name == "generate":
            logger.info("    ✍️ 执行文本生成/总结/推理任务...")
            
            # 尝试从规划任务中提取出用户传递的图片路径
            img_match = re.search(r"【多模态提示】：用户勾选了一张本地图表 \((.*?)\)", state.get("task_input", ""))
            selected_img_path = img_match.group(1).strip() if img_match else None

            if selected_img_path and os.path.exists(selected_img_path):
                logger.info("    🖼️ 识别到用户选中的图表，正在组装 LangChain 多模态消息体...")
                
                # 对图片进行 Base64 编码
                with open(selected_img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                ext = selected_img_path.split('.')[-1].lower()
                mime_type = "image/png"
                if ext in ['jpg', 'jpeg']: mime_type = "image/jpeg"
                elif ext == 'webp': mime_type = "image/webp"

                # 兼顾原有的提示词拼接
                prompt_text = TEXT_GENERATION_PROMPT.format(
                    chat_history=state.get("chat_history", "无"),
                    current_step=current_step,
                    context=context_str
                )
                
                # 按照OpenAI兼容格式组装LangChain多模态content块
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
                
                messages = [HumanMessage(content=content_blocks)]
                
                try:
                    res = self.llm.invoke(messages)
                    output = res.content
                except Exception as e:
                    logger.error("多模态生成任务失败: %s", e)
                    output = f"【多模态推理失败】: {str(e)}"
                    
            else:
                # 无图片generate，原有的纯文本链条逻辑不变
                prompt_template = ChatPromptTemplate.from_template(TEXT_GENERATION_PROMPT)
                chain = prompt_template | self.llm

                # 传入模板需要的变量
                res = chain.invoke({
                    "chat_history": state.get("chat_history", "无"),
                    "current_step": current_step,
                    "context": context_str
                })
                output = res.content

        else:
            # print(f"    ⚠️ 警告: 未知工具 '{tool_name}'")
            logger.warning("    ⚠️ 警告: 未知工具 '%s'", tool_name)
            output = f"【系统提示】: 无法执行。Planner 分配了未注册的工具 '{tool_name}'。"

        # print(f"    步骤输出预览: {output[:100]}...\n")
        logger.info("     步骤输出：%s", output)
        display_step = "从历史经验中提取高价值结论" if tool_name == "memo_output" else current_step

        return {
            "step_history": [f"Step: {display_step}\nTool: {tool_name}\nResult: {output}"]
        }