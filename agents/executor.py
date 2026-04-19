import re

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config.logger import get_logger
from config.settings import Settings
from graph.state import AgentState
from prompts.executor_prompts import TEXT_GENERATION_PROMPT, TOOL_EXTRACTION_PROMPT
from tools.academic_writer_tool import AcademicWriterTool
from tools.arxiv_tool import ArxivSearchTool
from tools.literature_rag_tool import LiteratureRagTool
from tools.literature_reader_tool import LiteratureReaderTool
from tools.semantic_scholar_tool import SemanticScholarSearchTool
from utils.exceptions import (
    ToolExecutionError,
    ToolExecutionTimeout,
    VectorDBConnectionError,
)
from utils.image_utils import encode_image_to_data_url, unique_existing_image_paths

logger = get_logger()


class ExecutorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.tools = {
            "arxiv_search": ArxivSearchTool(),
            "semantic_scholar_search": SemanticScholarSearchTool(),
            "academic_write": AcademicWriterTool(),
            "literature_read": LiteratureReaderTool(),
            "literature_rag_search": LiteratureRagTool(),
        }

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Executor] Node ---")

        current_index = state["current_step_index"]
        current_step = state["plan"][current_index]
        tool_name = state["planned_tools"][current_index]

        feedback = ""
        eval_res = state.get("evaluation_result")
        if eval_res and not eval_res.get("passed"):
            feedback = (
                f"\n【上一次执行未通过，评估反馈】{eval_res.get('feedback')}\n"
                "请据此调整本次工具参数或生成逻辑。"
            )

        logger.info("正在执行步骤: %s", current_step)
        context_raw = "\n".join(state.get("step_history", []))
        context_str = context_raw if context_raw else "无"
        resource_context = state.get("resource_context", "无")
        state_update = {}

        if tool_name == "memo_output":
            logger.info("    正在从语义缓存中提取结果。")
            output = self._extract_result_payload(current_step)
            container = st.session_state.get("current_stream_container")
            if container:
                container.markdown(f"### 从历史经验中提取到高价值结论\n\n{output}")

        elif tool_name in self.tools:
            output, tool_state_update = self._run_registered_tool(
                state=state,
                tool_name=tool_name,
                current_step=current_step,
                context_str=context_str,
                resource_context=resource_context,
                feedback=feedback,
            )
            state_update.update(tool_state_update)

        elif tool_name == "generate":
            logger.info("    正在执行文本生成或推理步骤。")
            output = self._run_generate_step(
                state=state,
                current_step=current_step,
                context_str=context_str,
                resource_context=resource_context,
            )

        elif tool_name == "trigger_reviewer_loop":
            logger.info("    正在唤起 Reviewer 子图。")
            output = self._run_reviewer_subgraph(state=state, current_step=current_step)

        else:
            logger.warning("    分配到了未知工具: %s", tool_name)
            output = f"【系统提示】无法执行，未注册工具 '{tool_name}'。"

        logger.info("    步骤输出: %s", output)
        display_step = (
            "从历史经验中提取高价值结论" if tool_name == "memo_output" else current_step
        )
        state_update["step_history"] = [
            f"Step: {display_step}\nTool: {tool_name}\nResult: {output}"
        ]
        return state_update

    def _run_registered_tool(
        self,
        state: AgentState,
        tool_name: str,
        current_step: str,
        context_str: str,
        resource_context: str,
        feedback: str,
    ) -> tuple[str, dict]:
        tool = self.tools[tool_name]
        logger.info("    准备调用工具: [%s]", tool_name)

        prompt_template = ChatPromptTemplate.from_template(TOOL_EXTRACTION_PROMPT)
        chain = prompt_template | self.llm
        query_res = chain.invoke(
            {
                "original_task": state.get("task_input", ""),
                "resource_context": resource_context,
                "tool_name": tool_name,
                "tool_desc": getattr(tool, "prompt_spec", "")
                or getattr(tool, "description", "工具执行"),
                "current_step": current_step,
                "context": context_str,
                "feedback": feedback,
            }
        )
        tool_input = query_res.content.strip()
        logger.info("    解析出的工具参数: [%s]", tool_input)

        try:
            tool_result = tool.run(tool_input)
            output = f"【{tool_name} 执行结果】\n{tool_result}"
        except ToolExecutionTimeout as exc:
            logger.error("工具 %s 执行超时: %s", tool_name, exc)
            output = f"【{tool_name} 执行超时】请精简参数后重试。"
        except (ToolExecutionError, VectorDBConnectionError) as exc:
            logger.error("工具 %s 执行失败: %s", tool_name, exc)
            output = f"【{tool_name} 执行失败】{str(exc)}"
        except Exception as exc:
            logger.exception("工具 %s 发生未知错误", tool_name)
            output = f"【{tool_name} 系统错误】{str(exc)}"

        state_update = {}
        if tool_name == "literature_rag_search":
            prior_images = state.get("retrieved_image_paths", [])
            rag_images = getattr(tool, "last_retrieved_images", [])
            merged_images = unique_existing_image_paths([*prior_images, *rag_images])
            if merged_images:
                state_update["retrieved_image_paths"] = merged_images

        return output, state_update

    def _run_generate_step(
        self,
        state: AgentState,
        current_step: str,
        context_str: str,
        resource_context: str,
    ) -> str:
        stream_llm = Settings.get_llm(temperature=0.1, streaming=True)
        container = st.session_state.get("current_stream_container")
        output = ""

        selected_image_path = state.get("selected_image_path")
        retrieved_image_paths = state.get("retrieved_image_paths", [])
        multimodal_images = unique_existing_image_paths(
            [selected_image_path, *retrieved_image_paths]
        )

        if multimodal_images:
            logger.info("    检测到图片上下文，使用多模态输入生成答案。")
            prompt_text = TEXT_GENERATION_PROMPT.format(
                chat_history=state.get("chat_history", "无"),
                resource_context=resource_context,
                current_step=current_step,
                context=context_str,
            )

            content_blocks = [
                {
                    "type": "text",
                    "text": (
                        "请结合以下文本上下文与附带图片回答任务。"
                        "图片可能来自用户手动勾选，也可能来自文献 RAG 自动命中的图表。\n\n"
                        f"{prompt_text}"
                    ),
                }
            ]
            for image_path in multimodal_images[:3]:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_data_url(image_path)},
                    }
                )

            messages = [HumanMessage(content=content_blocks)]
            try:
                for chunk in stream_llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not content:
                        continue
                    output += content
                    if container:
                        container.markdown(f"### 正在进行多模态分析...\n\n{output} ▌")
                if container:
                    container.markdown(f"### 分析完毕\n\n{output}")
            except Exception as exc:
                logger.error("多模态生成失败: %s", exc)
                output = f"【多模态推理失败】{str(exc)}"
            return output

        prompt_template = ChatPromptTemplate.from_template(TEXT_GENERATION_PROMPT)
        chain = prompt_template | self.llm
        try:
            for chunk in chain.stream(
                {
                    "chat_history": state.get("chat_history", "无"),
                    "resource_context": resource_context,
                    "current_step": current_step,
                    "context": context_str,
                }
            ):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not content:
                    continue
                output += content
                if container:
                    container.markdown(f"### 正在总结归纳...\n\n{output} ▌")
            if container:
                container.markdown(f"### 总结归纳完毕\n\n{output}")
        except Exception as exc:
            output = f"【文本生成失败】{str(exc)}"
        return output

    def _run_reviewer_subgraph(self, state: AgentState, current_step: str) -> str:
        from graph.graph_builder import build_reviewer_graph

        draft_content = self._extract_review_draft(state)
        if not draft_content.strip():
            return "【trigger_reviewer_loop 执行失败】未找到可供审稿的草稿内容。"

        reviewer_prompt = (
            f"{state.get('task_input', '')}\n\n"
            f"当前自审步骤要求：{current_step}"
        ).strip()
        reviewer_state = {
            "current_function": "d",
            "user_prompt": reviewer_prompt,
            "draft_content": draft_content,
            "original_draft_content": draft_content,
            "feedback": "",
            "task_intent": "",
            "review_mode": "",
            "review_focus": "",
            "diff_content": "",
            "final_diff_content": "",
            "retry_count": 0,
            "max_retries": 3,
            "status": "",
            "chat_history": state.get("chat_history", ""),
            "final_answer": "",
        }

        final_state = build_reviewer_graph().invoke(reviewer_state)
        final_draft = final_state.get("draft_content", draft_content)
        reviewer_feedback = final_state.get("feedback", "")
        final_diff = final_state.get("final_diff_content", "")
        status = final_state.get("status", "")

        if status == "reject":
            return (
                "### 自审结果\n"
                f"Reviewer 判定为 `reject`。\n\n"
                f"**原因**：{reviewer_feedback}"
            )

        sections = [
            "### 自审结果",
            f"Reviewer 判定为 `{status or 'pass'}`。",
            "#### 修订后正文",
            final_draft,
        ]
        if final_diff:
            sections.extend(
                [
                    "#### 增量修改视图",
                    final_diff,
                ]
            )
        if reviewer_feedback:
            sections.extend(
                [
                    "#### Reviewer 意见",
                    reviewer_feedback,
                ]
            )
        return "\n\n".join(sections)

    @staticmethod
    def _extract_result_payload(raw_text: str) -> str:
        match = re.search(r"Result:\s*(.*)", raw_text, re.DOTALL)
        return match.group(1).strip() if match else raw_text.strip()

    def _extract_review_draft(self, state: AgentState) -> str:
        step_history = state.get("step_history", [])
        if step_history:
            latest_payload = self._extract_result_payload(step_history[-1])
            latest_payload = re.sub(r"^【[^】]+】\n?", "", latest_payload).strip()
            if latest_payload:
                return latest_payload
        return state.get("task_input", "")
