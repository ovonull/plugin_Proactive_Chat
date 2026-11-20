"""拟人主动聊天：群里或私聊沉默到点，自己判断要不要开口并主动说一句。

收到真人消息时记录时间、归零未回复计数；巡检任务到点找出沉默会话，
先评分再按概率决定是否开口，连续若干次没人理就暂停，等用户开口再恢复。
"""

from __future__ import annotations

import random
from typing import Any

from app.services.plugin_sdk import PluginCallError, on_message, on_schedule

from proactive_ai import (
    ProactiveAiError,
    extract_sender,
    format_history,
    generate_with_retry,
    motivation_prompt,
    parse_score,
    score_prompt,
    SCORE_SYSTEM,
)
from proactive_runtime import (
    SCHEDULE_EVENT,
    config_bool,
    config_float,
    config_int,
    current_generation,
    load_state,
    next_generation,
    quiet_now,
    register_tick,
    safe_send,
    spawn,
    store_guard,
    update_state,
)
from proactive_state import (
    ProactiveStateError,
    ensure_session,
    get_session,
    mark_failed,
    mark_message,
    mark_proactive_sent,
    session_key,
    silence_seconds,
)

REQUIRED_CAPABILITIES = (
    "msg.send",
    "llm.generate",
    "plugin.storage",
    "scheduler.register",
)


def _has_required(host: Any) -> bool:
    """判断主动聊天必需能力是否齐全。"""
    return all(host.has(code) for code in REQUIRED_CAPABILITIES)


def _parse_history_response(response: Any) -> list[dict[str, Any]]:
    """从 get_*_msg_history 的返回里取出消息列表。

    兼容两种返回：data 直接是消息数组，或 data.messages 是数组。
    """
    if not isinstance(response, dict) or response.get("status") != "ok":
        return []
    data = response.get("data")
    rows: Any = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("messages") or []
    return [row for row in rows if isinstance(row, dict)]


async def _read_history(host: Any, chat_type: str, target_id: str) -> str:
    """读取会话最近几条聊天流水，读不到就当没有，不阻断主动流程。"""
    if not host.has("msg.history"):
        return ""
    action = "get_group_msg_history" if chat_type == "group" else "get_friend_msg_history"
    params: dict[str, Any] = {
        "limit": config_int(host, "history_limit", 12, 1, 30),
    }
    if chat_type == "group":
        params["group_id"] = int(target_id) if str(target_id).isdigit() else target_id
    else:
        params["user_id"] = int(target_id) if str(target_id).isdigit() else target_id
    try:
        response = await host.action(action, params)
    except PluginCallError as error:
        host.log(f"主动聊天读取历史失败，会话={chat_type}:{target_id}，错误={error}", "warning")
        return ""
    return format_history(_parse_history_response(response), str(host.bot_qq))


async def _touch_session(event: dict[str, Any], host: Any) -> None:
    """记录一次真人发言：写最后时间、归零未回复、解除暂停。"""
    chat_type = "group" if event.get("message_type") == "group" else "friend"
    target_id = str(
        event.get("group_id") if chat_type == "group" else event.get("user_id") or ""
    ).strip()
    sender_qq, _ = extract_sender(event)
    if not target_id or not sender_qq:
        return
    # 机器人自己的发言不算真人活跃，避免自言自语把计数洗掉。
    self_id = str(event.get("self_id") or host.bot_qq or "")
    if sender_qq == self_id:
        return

    def update(state: dict[str, Any]) -> tuple[bool, Any]:
        key = session_key(chat_type, target_id)
        session, _ = ensure_session(state, key, str(host.bot_qq), target_id)
        mark_message(session, sender_qq)
        return True, None

    await update_state(host, update)
    # 幂等登记巡检；调用方不持状态锁。
    await register_tick(host)


@on_message
async def handle_message(event: dict[str, Any], host: Any) -> None:
    """收到真人消息就刷新该会话的沉默起点，让主动计时重新来过。"""
    if event.get("post_type") != "message" or event.get("message_type") not in {
        "group",
        "private",
    }:
        return
    if not _has_required(host):
        return
    chat_type = "group" if event.get("message_type") == "group" else "friend"
    if chat_type == "group" and not config_bool(host, "group_chat_enabled", True):
        return
    if chat_type == "friend" and not config_bool(host, "friend_chat_enabled", True):
        return
    try:
        await _touch_session(event, host)
    except (PluginCallError, ProactiveStateError) as error:
        host.log(f"主动聊天记录发言失败：{error}", "error")


async def _should_speak(host: Any, chat_type: str, target_id: str, history: str) -> bool:
    """评分：此刻主动说一句是否自然；AI 异常时按不发言处理。"""
    if not config_bool(host, "score_enabled", True):
        return True
    unanswered = 0
    try:
        async with store_guard():
            state = await load_state(host)
            session = get_session(state, session_key(chat_type, target_id))
            if session is not None:
                unanswered = int(session.get("unanswered_count") or 0)
        _, speak = await generate_with_retry(
            host,
            score_prompt(history, unanswered),
            SCORE_SYSTEM,
            parse_score,
            operation="开口评分",
        )
        return bool(speak)
    except (PluginCallError, ProactiveAiError, ProactiveStateError) as error:
        host.log(f"主动聊天评分失败，按不发言处理，会话={chat_type}:{target_id}，错误={error}", "error")
        return False


def _probability_hit(host: Any, key: str, default: float) -> bool:
    """按配置的概率掷一次骰子；概率 ≥1 必真，≤0 必假。"""
    probability = config_float(host, key, default, 0.0, 1.0)
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return random.random() < probability


async def _run_one(host: Any, chat_type: str, target_id: str, generation: int) -> None:
    """对单个沉默会话走完整主动流程：评分 → 生成 → 发送 → 记数。"""
    if generation != current_generation():
        return
    key = session_key(chat_type, target_id)
    history = await _read_history(host, chat_type, target_id)
    if generation != current_generation():
        return
    if not await _should_speak(host, chat_type, target_id, history):
        host.log(f"主动聊天评分未通过，会话={key}", "info")
        return
    # 插话概率：评分通过不代表一定开口，按配置掷一次骰子。
    # 这是「随机插话」的关键——沉默到点但概率没中，就静默跳过本轮。
    if not _probability_hit(host, "speak_probability", 0.4):
        host.log(f"主动聊天插话概率未命中，本轮跳过，会话={key}", "info")
        return

    unanswered = 0
    try:
        async with store_guard():
            state = await load_state(host)
            session = get_session(state, key)
            if session is None or session.get("paused"):
                return
            unanswered = int(session.get("unanswered_count") or 0)
        prompt = motivation_prompt(host, chat_type, history, unanswered)
        text, _ = await generate_with_retry(
            host,
            prompt,
            "你是 QQ 聊天用户，只输出要发出去的那一句话，不要任何解释或动作描述。",
            lambda value: str(value or "").strip(),
            operation="生成主动消息",
        )
    except (PluginCallError, ProactiveAiError, ProactiveStateError) as error:
        host.log(f"主动聊天生成失败，会话={key}，错误={error}", "error")
        await _record_failure(host, key, str(error))
        return

    if generation != current_generation():
        return
    sent = await safe_send(host, chat_type, target_id, str(text or "").strip()[:1000])
    if not sent:
        await _record_failure(host, key, "主动消息发送失败")
        return

    max_unanswered = config_int(host, "max_unanswered", 3, 1, 20)
    try:

        def update(state: dict[str, Any]) -> tuple[bool, Any]:
            session = get_session(state, key)
            if session is None:
                return False, None
            mark_proactive_sent(session, max_unanswered)
            return True, None

        await update_state(host, update)
    except (PluginCallError, ProactiveStateError) as error:
        host.log(f"主动聊天记录发送结果失败，会话={key}，错误={error}", "error")
    host.log(f"主动聊天已发消息，会话={key}，累计未回复将到上限={max_unanswered}", "info")


async def _record_failure(host: Any, key: str, reason: str) -> None:
    """持久化本次失败原因；不改变暂停与未回复计数。"""
    try:

        def update(state: dict[str, Any]) -> tuple[bool, Any]:
            session = get_session(state, key)
            if session is None:
                return False, None
            mark_failed(session, reason)
            return True, None

        await update_state(host, update)
    except (PluginCallError, ProactiveStateError) as error:
        host.log(f"主动聊天记录失败原因也失败，会话={key}，错误={error}", "error")


async def _scan_sessions(host: Any) -> None:
    """巡检所有会话：挑出沉默够久、未停、未到上限的，逐个处理。"""
    try:
        async with store_guard():
            state = await load_state(host)
            sessions = [
                dict(item)
                for key, item in (state.get("sessions") or {}).items()
                if isinstance(item, dict) and ":" in str(key)
            ]
    except (PluginCallError, ProactiveStateError) as error:
        host.log(f"主动聊天巡检读状态失败：{error}", "error")
        return

    # 下限放到 1 分钟，方便装好后快速验证；默认值仍是 10 分钟。
    silence_minutes = config_int(host, "silence_minutes", 10, 1, 1440)
    threshold = silence_minutes * 60
    generation = current_generation()
    for session in sessions:
        if generation != current_generation():
            return
        if session.get("paused") or int(session.get("unanswered_count") or 0) >= config_int(
            host, "max_unanswered", 3, 1, 20
        ):
            continue
        if silence_seconds(session) < threshold:
            continue
        chat_type = "group" if session.get("chat_type") == "group" else "friend"
        if chat_type == "group" and not config_bool(host, "group_chat_enabled", True):
            continue
        if chat_type == "friend" and not config_bool(host, "friend_chat_enabled", True):
            continue
        target_id = str(session.get("target_id") or "").strip()
        if not target_id:
            continue
        spawn(
            host,
            _run_one(host, chat_type, target_id, generation),
            "生成主动消息",
        )


@on_schedule(SCHEDULE_EVENT)
async def handle_tick(event: dict[str, Any], host: Any) -> None:
    """宿主周期唤醒：每次递增代次，后台扫描并发送。"""
    if not _has_required(host):
        host.log("拟人主动聊天缺少必要能力，已跳过巡检", "error")
        return
    if quiet_now(host):
        host.log("当前为免打扰时段，跳过主动聊天巡检", "info")
        return
    next_generation()
    spawn(host, _scan_sessions(host), "主动聊天巡检")
