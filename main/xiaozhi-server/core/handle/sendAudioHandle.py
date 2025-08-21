import json
import asyncio
import time
from core.providers.tts.dto.dto import SentenceType
from core.utils import textUtils

TAG = __name__


async def sendAudioMessage(conn, sentenceType, audios, text):
    # 发送句子开始消息
    conn.logger.bind(tag=TAG).info(f"发送音频消息: {sentenceType}, {text}")

    pre_buffer = False

    # === 关键补丁：第一次句子先发 start，开启前端播放态 ===
    if getattr(conn.tts, "tts_audio_first_sentence", False):
        try:
            conn.client_is_speaking = True  # 明确标记正在说话，避免被打断
            await send_tts_message(conn, "start", None)
            conn.logger.bind(tag="TTSDEBUG").info("[MSG] -> sent 'start' (first sentence)")
        except Exception as e:
            conn.logger.bind(tag="TTSDEBUG").error(f"[MSG] send 'start' failed: {e}")

    if conn.tts.tts_audio_first_sentence:
        conn.logger.bind(tag=TAG).info(f"发送第一段语音: {text}")
        conn.tts.tts_audio_first_sentence = False
        pre_buffer = True

    # 告知这一句要播什么文本（UI 用）
    await send_tts_message(conn, "sentence_start", text)
    conn.logger.bind(tag="TTSDEBUG").info("[MSG] -> sent 'sentence_start'")

    # 真正发送音频帧
    await sendAudio(conn, audios, pre_buffer)

    # 发送结束消息（如果是最后一个文本）
    if conn.llm_finish_task and sentenceType == SentenceType.LAST:
        await send_tts_message(conn, "stop", None)
        conn.logger.bind(tag="TTSDEBUG").info("[MSG] -> sent 'stop'")
        conn.client_is_speaking = False
        if conn.close_after_chat:
            await conn.close()



# 播放音频
async def sendAudio(conn, audios, pre_buffer=True):
    if audios is None or len(audios) == 0:
        conn.logger.bind(tag="TTSDEBUG").warning("[AUDIO] no frames to send")
        return

    frame_duration = 60  # 每帧时长 ms（配合 Opus）
    start_time = time.perf_counter()
    play_position = 0

    # 预缓冲统计
    total_frames = len(audios)
    total_bytes = sum(len(p) for p in audios)
    conn.logger.bind(tag="TTSDEBUG").info(
        f"[AUDIO] begin sending frames={total_frames} bytes={total_bytes} pre_buffer={pre_buffer}"
    )

    # 仅当第一句话时执行预缓冲
    if pre_buffer:
        pre_buffer_frames = min(3, len(audios))
        pre_bytes = sum(len(audios[i]) for i in range(pre_buffer_frames))
        conn.logger.bind(tag="TTSDEBUG").info(
            f"[AUDIO] prebuffer frames={pre_buffer_frames} bytes={pre_bytes}"
        )
        for i in range(pre_buffer_frames):
            try:
                await conn.websocket.send(audios[i])
            except Exception as e:
                conn.logger.bind(tag="TTSDEBUG").error(f"[AUDIO] prebuffer send failed at {i}: {e}")
        remaining_audios = audios[pre_buffer_frames:]
    else:
        remaining_audios = audios

    sent = 0
    sent_bytes = 0
    for idx, opus_packet in enumerate(remaining_audios):
        if conn.client_abort:
            conn.logger.bind(tag="TTSDEBUG").warning("[AUDIO] aborted while sending")
            break

        # 重置无声超时
        conn.last_activity_time = time.time() * 1000

        expected_time = start_time + (play_position / 1000)
        current_time = time.perf_counter()
        delay = expected_time - current_time
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            await conn.websocket.send(opus_packet)
            sent += 1
            sent_bytes += len(opus_packet or b"")
        except Exception as e:
            conn.logger.bind(tag="TTSDEBUG").error(f"[AUDIO] send failed at idx={idx}: {e}")
            # 出错继续尝试下一帧
        finally:
            play_position += frame_duration

        # 适度采样日志：前 2 帧、每 20 帧、最后 1 帧
        if idx < 2 or (idx % 20 == 0) or (idx == len(remaining_audios) - 1):
            conn.logger.bind(tag="TTSDEBUG").info(
                f"[AUDIO] progress idx={idx+1}/{len(remaining_audios)} sent={sent} bytes={sent_bytes}"
            )

    conn.logger.bind(tag="TTSDEBUG").info(
        f"[AUDIO] done total_sent={sent + (3 if pre_buffer else 0)} approx_bytes={sent_bytes}"
    )



async def send_tts_message(conn, state, text=None):
    """发送 TTS 状态消息"""
    message = {"type": "tts", "state": state, "session_id": conn.session_id}
    if text is not None:
        message["text"] = textUtils.check_emoji(text)

    # TTS播放结束
    if state == "stop":
        # 播放提示音
        tts_notify = conn.config.get("enable_stop_tts_notify", False)
        if tts_notify:
            stop_tts_notify_voice = conn.config.get(
                "stop_tts_notify_voice", "config/assets/tts_notify.mp3"
            )
            audios, _ = conn.tts.audio_to_opus_data(stop_tts_notify_voice)
            await sendAudio(conn, audios)
        # 清除服务端讲话状态
        conn.clearSpeakStatus()

    # 发送消息到客户端
    await conn.websocket.send(json.dumps(message))


async def send_stt_message(conn, text):
    end_prompt_str = conn.config.get("end_prompt", {}).get("prompt")
    if end_prompt_str and end_prompt_str == text:
        await send_tts_message(conn, "start")
        return

    # 解析 JSON 文本显示
    display_text = text
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                display_text = parsed_data["content"]
                if "speaker" in parsed_data:
                    conn.current_speaker = parsed_data["speaker"]
    except (json.JSONDecodeError, TypeError):
        display_text = text

    stt_text = textUtils.get_string_no_punctuation_or_emoji(display_text)
    try:
        await conn.websocket.send(
            json.dumps({"type": "stt", "text": stt_text, "session_id": conn.session_id})
        )
        conn.logger.bind(tag="TTSDEBUG").info(
            f"[STT] echo text.len={len(stt_text or '')}"
        )
    except Exception as e:
        conn.logger.bind(tag="TTSDEBUG").error(f"[STT] websocket send failed: {e}")

    conn.client_is_speaking = True
    await send_tts_message(conn, "start")