"""设备端MCP客户端支持模块"""

import json
import asyncio
import re
from concurrent.futures import Future
from core.utils.util import get_vision_url, sanitize_tool_name
from core.utils.auth import AuthToken
from config.logger import setup_logging
from core.state_registry import set_state,get_state
from core.handle.sendAudioHandle import send_tts_message
import time
from core.providers.tts.dto.dto import TTSMessageDTO, ContentType, SentenceType as TTSSentenceType
import uuid



TAG = __name__
logger = setup_logging()


class MCPClient:
    """设备端MCP客户端，用于管理MCP状态和工具"""

    def __init__(self):
        self.tools = {}  # sanitized_name -> tool_data
        self.name_mapping = {}
        self.ready = False
        self.call_results = {}  # To store Futures for tool call responses
        self.next_id = 1
        self.lock = asyncio.Lock()
        self._cached_available_tools = None  # Cache for get_available_tools

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_available_tools(self) -> list:
        # Check if the cache is valid
        if self._cached_available_tools is not None:
            return self._cached_available_tools

        # If cache is not valid, regenerate the list
        result = []
        for tool_name, tool_data in self.tools.items():
            function_def = {
                "name": tool_name,
                "description": tool_data["description"],
                "parameters": {
                    "type": tool_data["inputSchema"].get("type", "object"),
                    "properties": tool_data["inputSchema"].get("properties", {}),
                    "required": tool_data["inputSchema"].get("required", []),
                },
            }
            result.append({"type": "function", "function": function_def})

        self._cached_available_tools = result  # Store the generated list in cache
        return result

    async def is_ready(self) -> bool:
        async with self.lock:
            return self.ready

    async def set_ready(self, status: bool):
        async with self.lock:
            self.ready = status

    async def add_tool(self, tool_data: dict):
        async with self.lock:
            sanitized_name = sanitize_tool_name(tool_data["name"])
            self.tools[sanitized_name] = tool_data
            self.name_mapping[sanitized_name] = tool_data["name"]
            self._cached_available_tools = (
                None  # Invalidate the cache when a tool is added
            )

    async def get_next_id(self) -> int:
        async with self.lock:
            current_id = self.next_id
            self.next_id += 1
            return current_id

    async def register_call_result_future(self, id: int, future: Future):
        async with self.lock:
            self.call_results[id] = future

    async def resolve_call_result(self, id: int, result: any):
        async with self.lock:
            if id in self.call_results:
                future = self.call_results.pop(id)
                if not future.done():
                    future.set_result(result)

    async def reject_call_result(self, id: int, exception: Exception):
        async with self.lock:
            if id in self.call_results:
                future = self.call_results.pop(id)
                if not future.done():
                    future.set_exception(exception)

    async def cleanup_call_result(self, id: int):
        async with self.lock:
            if id in self.call_results:
                self.call_results.pop(id)
    
async def handle_encourage(conn, params: dict):
    # 仅限训练中
    if get_state(conn, "idle") != "training":
        conn.logger.bind(tag="MCP").info("[ENCOURAGE] ignore: not in training")
        return

    # 准备一点上下文（可选）
    set_order = int(params.get("set", 0) or 0)
    rep_index = int(params.get("rep", 0) or 0)
    total     = int(params.get("total", 0) or 0)
    ex_id     = int(params.get("exercise", -1) or -1)

    # 用 LLM 生成一句非常短的鼓励
    # 这里直接走 conn.chat，避免 startToChat 的聊天闸门
    # 给一条“用户指令”式文本，明确长度与风格
    instruct = f"【简短鼓励】已完成{rep_index}/{total}。只用一两句，口令风，避免寒暄。"
    conn.executor.submit(conn.chat, instruct)


async def send_mcp_message(conn, payload: dict):
    """Helper to send MCP messages, encapsulating common logic."""
    if not conn.features.get("mcp"):
        logger.bind(tag=TAG).warning("客户端不支持MCP，无法发送MCP消息")
        return

    message = json.dumps({"type": "mcp", "payload": payload})

    try:
        await conn.websocket.send(message)
        logger.bind(tag=TAG).info(f"成功发送MCP消息: {message}")
    except Exception as e:
        logger.bind(tag=TAG).error(f"发送MCP消息失败: {e}")


async def handle_mcp_message(conn, mcp_client: MCPClient, payload: dict):
    """处理MCP消息,包括初始化、工具列表和工具调用响应等"""
    logger.bind(tag=TAG).info(f"处理MCP消息: {str(payload)[:100]}")

    if not isinstance(payload, dict):
        logger.bind(tag=TAG).error("MCP消息缺少payload字段或格式错误")
        return

    # Handle result
    if "result" in payload:
        result = payload["result"]
        msg_id = int(payload.get("id", 0))

        # Check for tool call response first
        if msg_id in mcp_client.call_results:
            logger.bind(tag=TAG).debug(
                f"收到工具调用响应，ID: {msg_id}, 结果: {result}"
            )
            await mcp_client.resolve_call_result(msg_id, result)
            return

        if msg_id == 1:  # mcpInitializeID
            logger.bind(tag=TAG).debug("收到MCP初始化响应")
            server_info = result.get("serverInfo")
            if isinstance(server_info, dict):
                name = server_info.get("name")
                version = server_info.get("version")
                logger.bind(tag=TAG).info(
                    f"客户端MCP服务器信息: name={name}, version={version}"
                )
            return

        elif msg_id == 2:  # mcpToolsListID
            logger.bind(tag=TAG).debug("收到MCP工具列表响应")
            if isinstance(result, dict) and "tools" in result:
                tools_data = result["tools"]
                if not isinstance(tools_data, list):
                    logger.bind(tag=TAG).error("工具列表格式错误")
                    return

                logger.bind(tag=TAG).info(
                    f"客户端设备支持的工具数量: {len(tools_data)}"
                )

                for i, tool in enumerate(tools_data):
                    if not isinstance(tool, dict):
                        continue

                    name = tool.get("name", "")
                    description = tool.get("description", "")
                    input_schema = {"type": "object", "properties": {}, "required": []}

                    if "inputSchema" in tool and isinstance(tool["inputSchema"], dict):
                        schema = tool["inputSchema"]
                        input_schema["type"] = schema.get("type", "object")
                        input_schema["properties"] = schema.get("properties", {})
                        input_schema["required"] = [
                            s for s in schema.get("required", []) if isinstance(s, str)
                        ]

                    new_tool = {
                        "name": name,
                        "description": description,
                        "inputSchema": input_schema,
                    }
                    await mcp_client.add_tool(new_tool)
                    logger.bind(tag=TAG).debug(f"客户端工具 #{i+1}: {name}")

                # 替换所有工具描述中的工具名称
                for tool_data in mcp_client.tools.values():
                    if "description" in tool_data:
                        description = tool_data["description"]
                        # 遍历所有工具名称进行替换
                        for (
                            sanitized_name,
                            original_name,
                        ) in mcp_client.name_mapping.items():
                            description = description.replace(
                                original_name, sanitized_name
                            )
                        tool_data["description"] = description

                next_cursor = result.get("nextCursor", "")
                if next_cursor:
                    logger.bind(tag=TAG).info(f"有更多工具，nextCursor: {next_cursor}")
                    await send_mcp_tools_list_continue_request(conn, next_cursor)
                else:
                    await mcp_client.set_ready(True)
                    logger.bind(tag=TAG).info("所有工具已获取，MCP客户端准备就绪")

                    # 刷新工具缓存，确保MCP工具被包含在函数列表中
                    if hasattr(conn, "func_handler") and conn.func_handler:
                        conn.func_handler.tool_manager.refresh_tools()
                        conn.func_handler.current_support_functions()
            return

    # Handle method calls (requests from the client)
    elif "method" in payload:
        method = str(payload.get("method", "")).strip()
        logger.bind(tag=TAG).info(f"收到MCP客户端请求: {method}")

      # === A) 统一设态（兼容多种别名）===
        if method in ("notifications/state_set", "notification/set_state", "state_set", "set_state"):
            params = payload.get("params") or {}
            mode = (params.get("mode") or "").lower()
            if mode in ("idle", "training", "rest"):
                old = get_state(conn, "idle")
                set_state(conn, mode)
                logger.bind(tag=TAG).info(
                    f"[MCP][STATE] device={getattr(conn,'device_id',None)} {old} -> {mode} by state_set"
                )
            else:
                logger.bind(tag=TAG).warning(f"[MCP][STATE] ignore invalid mode: {mode}")
            return
        
        elif method == "notifications/encourage":
            params = payload.get("params") or {}
            logger.bind(tag=TAG).info(f"[MCP][ENCOURAGE] params={params}")

            # 仅在 training 态播报
            if get_state(conn, "idle") != "training":
                logger.bind(tag=TAG).info("[MCP][ENCOURAGE] skip: not in training")
                return

            # 生成一句极短鼓励
            sys_prompt = "你是小智。生成一句极短的训练鼓励（<= 12字），口令风，避免寒暄和标点滥用。"
            set_ = params.get("set"); rep = params.get("rep"); total = params.get("total")
            exercise = params.get("exercise"); phase = params.get("phase")
            if rep and total:
                user_prompt = f"第{set_ or '?'}组 {rep}/{total} 次，动作:{exercise or '?'}，阶段:{phase or 'late'}。来一句鼓励。"
            else:
                user_prompt = "训练中。来一句极短鼓励。"

            try:
                text = (conn.llm.response_no_stream(sys_prompt, user_prompt) or "").strip()
            except Exception as e:
                logger.bind(tag=TAG).warning(f"[MCP][ENCOURAGE] llm gen failed: {e}")
                text = "稳住，别晃！"
            if len(text) > 16:
                text = text[:16]

            # —— 保护窗：避免被 VAD/ASR 误触或打断 —— #
            now_ms = int(time.time() * 1000)
            conn.encourage_playing = True
            # 估个时长：~ 250ms/字 + 500ms 余量
            est_ms = int(len(text) * 250 + 500)
            conn.encourage_until_ms = now_ms + max(1500, est_ms)
            conn.last_activity_time = now_ms

            # 若此刻在说话，先止住上一段
            if getattr(conn, "client_is_speaking", False):
                await send_tts_message(conn, "stop", None)
                conn.client_is_speaking = False

            # 关键：先发 start 让设备切到 Speaking
            await send_tts_message(conn, "start", None)

            # 准备一次独立的句子会话
            conn.sentence_id = uuid.uuid4().hex
            conn.tts.tts_audio_first_sentence = True

            # FIRST → MIDDLE(TEXT) → LAST
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail=text,
                )
            )
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )

            logger.bind(tag=TAG).info(f"[MCP][ENCOURAGE] speak: {text}")
            return
        
        elif method == "notifications/weight_delta":
            params = payload.get("params") or {}
            delta = params.get("delta")
            unit  = (params.get("unit") or "kg").lower()

            # 解析 delta
            try:
                delta = float(delta)
            except Exception:
                logger.bind(tag=TAG).warning(f"[MCP][WEIGHT] invalid delta: {delta!r}")
                return

            # 组播文案（两位小数，kg → 千克）
            direction = "增加" if delta > 0 else "减少"
            unit_cn = "千克" if unit == "kg" else unit
            say = f"{direction}{abs(delta):.2f}{unit_cn}"

            # 若此刻在说话，先止住上一段（可选）
            if getattr(conn, "client_is_speaking", False):
                await send_tts_message(conn, "stop", None)
                conn.client_is_speaking = False

            # 起播：先发 start，再走 FIRST→MIDDLE(TEXT)→LAST
            await send_tts_message(conn, "start", None)

            conn.sentence_id = uuid.uuid4().hex
            conn.tts.tts_audio_first_sentence = True   # 触发首句预缓冲
            conn.last_activity_time = int(time.time() * 1000)

            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=TTSSentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=TTSSentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail=say,
                )
            )
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=TTSSentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )

            logger.bind(tag=TAG).info(f"[MCP][WEIGHT] speak: {say}")
            return



    elif "error" in payload:
        error_data = payload["error"]
        error_msg = error_data.get("message", "未知错误")
        logger.bind(tag=TAG).error(f"收到MCP错误响应: {error_msg}")

        msg_id = int(payload.get("id", 0))
        if msg_id in mcp_client.call_results:
            await mcp_client.reject_call_result(
                msg_id, Exception(f"MCP错误: {error_msg}")
            )


async def send_mcp_initialize_message(conn):
    """发送MCP初始化消息"""

    vision_url = get_vision_url(conn.config)

    # 密钥生成token
    auth = AuthToken(conn.config["server"]["auth_key"])
    token = auth.generate_token(conn.headers.get("device-id"))

    vision = {
        "url": vision_url,
        "token": token,
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,  # mcpInitializeID
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
                "vision": vision,
            },
            "clientInfo": {
                "name": "XiaozhiClient",
                "version": "1.0.0",
            },
        },
    }
    logger.bind(tag=TAG).info("发送MCP初始化消息")
    await send_mcp_message(conn, payload)


async def send_mcp_tools_list_request(conn):
    """发送MCP工具列表请求"""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,  # mcpToolsListID
        "method": "tools/list",
    }
    logger.bind(tag=TAG).debug("发送MCP工具列表请求")
    await send_mcp_message(conn, payload)


async def send_mcp_tools_list_continue_request(conn, cursor: str):
    """发送带有cursor的MCP工具列表请求"""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,  # mcpToolsListID (same ID for continuation)
        "method": "tools/list",
        "params": {"cursor": cursor},
    }
    logger.bind(tag=TAG).info(f"发送带cursor的MCP工具列表请求: {cursor}")
    await send_mcp_message(conn, payload)


async def call_mcp_tool(
    conn, mcp_client: MCPClient, tool_name: str, args: str = "{}", timeout: int = 30
):
    """
    调用指定的工具，并等待响应
    """
    if not await mcp_client.is_ready():
        raise RuntimeError("MCP客户端尚未准备就绪")

    if not mcp_client.has_tool(tool_name):
        raise ValueError(f"工具 {tool_name} 不存在")

    tool_call_id = await mcp_client.get_next_id()
    result_future = asyncio.Future()
    await mcp_client.register_call_result_future(tool_call_id, result_future)

    # 处理参数
    try:
        if isinstance(args, str):
            # 确保字符串是有效的JSON
            if not args.strip():
                arguments = {}
            else:
                try:
                    # 尝试直接解析
                    arguments = json.loads(args)
                except json.JSONDecodeError:
                    # 如果解析失败，尝试合并多个JSON对象
                    try:
                        # 使用正则表达式匹配所有JSON对象
                        json_objects = re.findall(r"\{[^{}]*\}", args)
                        if len(json_objects) > 1:
                            # 合并所有JSON对象
                            merged_dict = {}
                            for json_str in json_objects:
                                try:
                                    obj = json.loads(json_str)
                                    if isinstance(obj, dict):
                                        merged_dict.update(obj)
                                except json.JSONDecodeError:
                                    continue
                            if merged_dict:
                                arguments = merged_dict
                            else:
                                raise ValueError(f"无法解析任何有效的JSON对象: {args}")
                        else:
                            raise ValueError(f"参数JSON解析失败: {args}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"参数JSON解析失败: {str(e)}, 原始参数: {args}"
                        )
                        raise ValueError(f"参数JSON解析失败: {str(e)}")
        elif isinstance(args, dict):
            arguments = args
        else:
            raise ValueError(f"参数类型错误，期望字符串或字典，实际类型: {type(args)}")

        # 确保参数是字典类型
        if not isinstance(arguments, dict):
            raise ValueError(f"参数必须是字典类型，实际类型: {type(arguments)}")

    except Exception as e:
        if not isinstance(e, ValueError):
            raise ValueError(f"参数处理失败: {str(e)}")
        raise e

    actual_name = mcp_client.name_mapping.get(tool_name, tool_name)
    payload = {
        "jsonrpc": "2.0",
        "id": tool_call_id,
        "method": "tools/call",
        "params": {"name": actual_name, "arguments": arguments},
    }

    logger.bind(tag=TAG).info(f"发送客户端mcp工具调用请求: {actual_name}，参数: {args}")
    await send_mcp_message(conn, payload)

    try:
        # Wait for response or timeout
        raw_result = await asyncio.wait_for(result_future, timeout=timeout)
        logger.bind(tag=TAG).info(
            f"客户端mcp工具调用 {actual_name} 成功，原始结果: {raw_result}"
        )

        if isinstance(raw_result, dict):
            if raw_result.get("isError") is True:
                error_msg = raw_result.get(
                    "error", "工具调用返回错误，但未提供具体错误信息"
                )
                raise RuntimeError(f"工具调用错误: {error_msg}")

            content = raw_result.get("content")
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    # 直接返回文本内容，不进行JSON解析
                    return content[0]["text"]
        # 如果结果不是预期的格式，将其转换为字符串
        return str(raw_result)
    except asyncio.TimeoutError:
        await mcp_client.cleanup_call_result(tool_call_id)
        raise TimeoutError("工具调用请求超时")
    except Exception as e:
        await mcp_client.cleanup_call_result(tool_call_id)
        raise e
