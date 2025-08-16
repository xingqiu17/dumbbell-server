from core.handle.sendAudioHandle import send_stt_message
from core.handle.intentHandler import handle_user_intent
from core.utils.output_counter import check_device_output_limit
from core.handle.abortHandle import handleAbortMessage
import time
import asyncio
import json
import re
from core.handle.sendAudioHandle import SentenceType
from core.utils.util import audio_to_data
from core.state_registry import get_state
from core.state_registry import set_state
from core.handle.sendAudioHandle import send_tts_message
from core.providers.tools.device_mcp.mcp_handler import call_mcp_tool


TAG = __name__

def _encourage_window_active(conn) -> bool:
    now_ms = int(time.time() * 1000)
    return bool(getattr(conn, "encourage_playing", False) and getattr(conn, "encourage_until_ms", 0) > now_ms)


def with_state_system_note(conn, messages: list):
    mode = get_state(conn, "idle")  # idle | training | rest
    return [{"role":"system","content": f"[state] {mode}"}] + messages

async def handleAudioMessage(conn, audio):

     # === 状态闸门：在 training/rest 不进行聆听（不做VAD、不做ASR、不打断TTS）===
    try:
        mode = get_state(conn, "idle")
    except Exception:
        mode = "idle"
    if mode in ("training", "rest"):
        await no_voice_close_connect(conn, False)
        return
    # 当前片段是否有人说话
    have_voice = conn.vad.is_vad(conn, audio)

    # 如果设备刚刚被唤醒，短暂忽略VAD检测
    if have_voice and hasattr(conn, "just_woken_up") and conn.just_woken_up:
        have_voice = False
        conn.asr_audio.clear()
        if not hasattr(conn, "vad_resume_task") or conn.vad_resume_task.done():
            conn.vad_resume_task = asyncio.create_task(resume_vad_detection(conn))
        return

    # === 新增：鼓励保护窗内，忽略这段“有声”，避免打断 & 避免把TTS吸进ASR ===
    if have_voice and _encourage_window_active(conn):
        conn.logger.bind(tag=TAG).debug("[ENCOURAGE] window active: ignore voice chunk (no abort, no ASR)")
        have_voice = False
        # 可选：清空一次缓存，防止误触发
        conn.asr_audio.clear()

    # 仅在不处于保护窗时才允许打断
    if have_voice:
        if conn.client_is_speaking and not _encourage_window_active(conn):
            await handleAbortMessage(conn)

    # 设备长时间空闲检测，用于say goodbye
    await no_voice_close_connect(conn, have_voice)
    # 接收音频（在保护窗内 have_voice=False，不会被ASR作为“有效语音段”处理）
    await conn.asr.receive_audio(conn, audio, have_voice)



async def resume_vad_detection(conn):
    # 等待2秒后恢复VAD检测
    await asyncio.sleep(1)
    conn.just_woken_up = False


_NORM_PUNC = r"[，。！？!.?\s]"
def _norm_text(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(_NORM_PUNC, "", t)
    return t

_EXIT_RE = re.compile(r"(退出|停止|结束)(训练|本次训练)|stoptraining|end(workout|training)")
def _looks_exit_training(t: str) -> bool:
    return bool(_EXIT_RE.search(_norm_text(t)))

_SKIP_RE = re.compile(r"(跳过|略过|不用|结束)休息|继续(训练|下一组)|开始下一组|skiprest|nextset")
def _looks_skip_rest(t: str) -> bool:
    return bool(_SKIP_RE.search(_norm_text(t)))

async def _mcp_call(conn, sanitized_tool_name: str) -> bool:
    """调用设备端 MCP 工具（传入 sanitize 后的名字）。成功返回 True。"""
    mc = getattr(conn, "mcp_client", None)
    if not mc:
        conn.logger.info("[MCP] no mcp_client on conn")
        return False
    if not await mc.is_ready():
        conn.logger.info("[MCP] mcp_client not ready")
        return False
    if not mc.has_tool(sanitized_tool_name):
        # 打印一下有哪些可用工具便于排查
        conn.logger.info(f"[MCP] tool not found: {sanitized_tool_name}; available={list(mc.tools.keys())}")
        return False

    try:
        res = await call_mcp_tool(conn, mc, sanitized_tool_name, args="{}")
        conn.logger.info(f"[MCP] tools/call ok: {sanitized_tool_name}, result={res}")
        return True
    except Exception as e:
        conn.logger.warning(f"[MCP] tools/call failed: {sanitized_tool_name}, err={e}")
        return False

async def startToChat(conn, text):
    # 检查输入是否是JSON格式（包含说话人信息）
    speaker_name = None
    actual_text = text
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            data = json.loads(text)
            if 'speaker' in data and 'content' in data:
                speaker_name = data['speaker']
                actual_text = data['content']
                conn.logger.bind(tag=TAG).info(f"解析到说话人信息: {speaker_name}")
    except (json.JSONDecodeError, KeyError):
        pass

    conn.current_speaker = speaker_name if speaker_name else None

    if conn.need_bind:
        await check_bind_device(conn)
        return

    # 限流 & 抢占说话者
    if conn.max_output_size > 0:
        if check_device_output_limit(conn.headers.get("device-id"), conn.max_output_size):
            await max_out_size(conn)
            return
    if conn.client_is_speaking and not _encourage_window_active(conn):
        await handleAbortMessage(conn)

        # === 1) 取设备状态 ===
    mode = get_state(conn, "idle")  # idle | training | rest

    # === 1.1) 训练/休息：硬静音（不做任何识别/意图/LLM），直接忽略 ===
    if mode in ("training", "rest"):
        conn.logger.bind(tag=TAG).info(f"[CHAT_GATE] hard mute: ignore ALL user text in {mode}")
        return

    # === 2) idle：允许意图与对话 ===
    state_prefixed_text = f"[state] {mode}\n{actual_text}"
    intent_handled = await handle_user_intent(conn, state_prefixed_text)
    conn.logger.bind(tag=TAG).info(f"[INTENT] handled={intent_handled} mode={mode} text={actual_text}")
    if intent_handled:
        return

    # === 5) idle：放行给 LLM ===
    await send_stt_message(conn, actual_text)
    conn.logger.bind(tag=TAG).info("[CHAT_GATE] pass to LLM (idle)")
    conn.executor.submit(conn.chat, actual_text)


async def no_voice_close_connect(conn, have_voice):
    if have_voice:
        conn.last_activity_time = time.time() * 1000
        return
        # 训练/休息态下直接跳过超时结束话术
    try:
        mode = get_state(conn, "idle")
    except Exception:
        mode = "idle"
    if mode in ("training", "rest"):
        return

    # 只有在已经初始化过时间戳的情况下才进行超时检查
    if conn.last_activity_time > 0.0:
        no_voice_time = time.time() * 1000 - conn.last_activity_time
        close_connection_no_voice_time = int(
            conn.config.get("close_connection_no_voice_time", 120)
        )
        if (
            not conn.close_after_chat
            and no_voice_time > 1000 * close_connection_no_voice_time
        ):
            conn.close_after_chat = True
            conn.client_abort = False
            end_prompt = conn.config.get("end_prompt", {})
            if end_prompt and end_prompt.get("enable", True) is False:
                conn.logger.bind(tag=TAG).info("结束对话，无需发送结束提示语")
                await conn.close()
                return
            prompt = end_prompt.get("prompt")
            if not prompt:
                prompt = "请你以```时间过得真快```未来头，用富有感情、依依不舍的话来结束这场对话吧。！"
            await startToChat(conn, prompt)


async def max_out_size(conn):
    text = "不好意思，我现在有点事情要忙，明天这个时候我们再聊，约好了哦！明天不见不散，拜拜！"
    await send_stt_message(conn, text)
    file_path = "config/assets/max_output_size.wav"
    opus_packets, _ = audio_to_data(file_path)
    conn.tts.tts_audio_queue.put((SentenceType.LAST, opus_packets, text))
    conn.close_after_chat = True


async def check_bind_device(conn):
    if conn.bind_code:
        # 确保bind_code是6位数字
        if len(conn.bind_code) != 6:
            conn.logger.bind(tag=TAG).error(f"无效的绑定码格式: {conn.bind_code}")
            text = "绑定码格式错误，请检查配置。"
            await send_stt_message(conn, text)
            return

        text = f"请登录控制面板，输入{conn.bind_code}，绑定设备。"
        await send_stt_message(conn, text)

        # 播放提示音
        music_path = "config/assets/bind_code.wav"
        opus_packets, _ = audio_to_data(music_path)
        conn.tts.tts_audio_queue.put((SentenceType.FIRST, opus_packets, text))

        # 逐个播放数字
        for i in range(6):  # 确保只播放6位数字
            try:
                digit = conn.bind_code[i]
                num_path = f"config/assets/bind_code/{digit}.wav"
                num_packets, _ = audio_to_data(num_path)
                conn.tts.tts_audio_queue.put((SentenceType.MIDDLE, num_packets, None))
            except Exception as e:
                conn.logger.bind(tag=TAG).error(f"播放数字音频失败: {e}")
                continue
        conn.tts.tts_audio_queue.put((SentenceType.LAST, [], None))
    else:
        text = f"没有找到该设备的版本信息，请正确配置 OTA地址，然后重新编译固件。"
        await send_stt_message(conn, text)
        music_path = "config/assets/bind_not_found.wav"
        opus_packets, _ = audio_to_data(music_path)
        conn.tts.tts_audio_queue.put((SentenceType.LAST, opus_packets, text))
