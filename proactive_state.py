"""主动聊天插件的会话状态：归一化、校验与查询。

状态存在插件自己的 storage.json 里，每个会话记最后发言时间、已主动次数、
暂停标记。所有写入都要带校验和，避免并发覆盖。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

MAX_SESSIONS = 500
MAX_UNANSWERED = 20
SCHEMA_VERSION = 1


class ProactiveStateError(ValueError):
    """状态数据不能用，或请求超出保护上限。"""


def session_key(chat_type: str, target_id: str) -> str:
    """生成插件内部的会话键，群和私聊分开存。"""
    kind = "group" if str(chat_type) == "group" else "friend"
    return f"{kind}:{str(target_id).strip()}"


def beijing_now() -> datetime:
    """取北京时间，和宿主全链路口径一致。"""
    return datetime.now(timezone(timedelta(hours=8)))


def format_time(value: datetime) -> str:
    """把时间写成固定格式，便于比对和人工查看。"""
    return value.astimezone(timezone(timedelta(hours=8))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def parse_time(value: Any) -> datetime:
    """按固定格式读回时间，坏值回到 Unix 纪元，不抛异常。"""
    try:
        return datetime.strptime(str(value or ""), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone(timedelta(hours=8))
        )
    except (TypeError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone(timedelta(hours=8)))


def _as_int(value: Any, default: int = 0) -> int:
    """把状态里的计数字段读成非负整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def normalize_session(raw: Any, key: str) -> dict[str, Any] | None:
    """归一化单条会话状态，字段缺失时给默认值。"""
    if not isinstance(raw, dict):
        return None
    chat_type, _, target_id = key.partition(":")
    return {
        "bot_qq": str(raw.get("bot_qq") or ""),
        "chat_type": "group" if chat_type == "group" else "friend",
        "target_id": target_id or str(raw.get("target_id") or ""),
        "last_message_at": format_time(parse_time(raw.get("last_message_at"))),
        "last_sender_qq": str(raw.get("last_sender_qq") or ""),
        "unanswered_count": min(_as_int(raw.get("unanswered_count")), MAX_UNANSWERED),
        "paused": bool(raw.get("paused", False)),
        "last_proactive_at": str(raw.get("last_proactive_at") or ""),
        "last_error": str(raw.get("last_error") or "")[:300],
    }


def normalize_state(raw: Any) -> dict[str, Any]:
    """归一化整份状态；坏键直接丢弃，保留校验和由调用方填写。"""
    source = raw if isinstance(raw, dict) else {}
    sessions: dict[str, Any] = {}
    raw_sessions = source.get("sessions")
    if isinstance(raw_sessions, dict):
        for key, value in raw_sessions.items():
            if not isinstance(key, str) or len(sessions) >= MAX_SESSIONS:
                continue
            normalized = normalize_session(value, key)
            if normalized is not None:
                sessions[key] = normalized
    return {
        "schema_version": SCHEMA_VERSION,
        "sessions": sessions,
        "_storage_checksum": str(source.get("_storage_checksum") or ""),
    }


def get_session(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    """读取一条会话状态，不存在返回 None。"""
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        return None
    value = sessions.get(key)
    return value if isinstance(value, dict) else None


def ensure_session(
    state: dict[str, Any], key: str, bot_qq: str, target_id: str
) -> tuple[dict[str, Any], bool]:
    """取得会话状态，没有就新建；返回 (状态, 是否新建)。"""
    current = get_session(state, key)
    if current is not None:
        return current, False
    sessions = state.setdefault("sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}
        state["sessions"] = sessions
    if len(sessions) >= MAX_SESSIONS:
        raise ProactiveStateError("主动聊天会话数已达上限")
    chat_type = "group" if key.startswith("group:") else "friend"
    created = normalize_session(
        {"bot_qq": bot_qq, "chat_type": chat_type, "target_id": target_id}, key
    )
    if created is None:  # pragma: no cover - 上面构造的一定合法
        raise ProactiveStateError("创建会话状态失败")
    sessions[key] = created
    return created, True


def silence_seconds(session: dict[str, Any]) -> int:
    """距离这条会话最后发言过了多少秒，未来时间按 0 处理。"""
    last = parse_time(session.get("last_message_at"))
    delta = (beijing_now() - last).total_seconds()
    return max(0, int(delta))


def mark_message(session: dict[str, Any], sender_qq: str) -> None:
    """记录一次真人发言，并归零未回复计数、解除暂停。"""
    session["last_message_at"] = format_time(beijing_now())
    session["last_sender_qq"] = str(sender_qq or "")
    session["unanswered_count"] = 0
    session["paused"] = False
    session["last_error"] = ""


def mark_proactive_sent(session: dict[str, Any], max_unanswered: int) -> None:
    """记一次主动发送，累计未回复次数；到上限就暂停。"""
    count = min(_as_int(session.get("unanswered_count")) + 1, MAX_UNANSWERED)
    session["unanswered_count"] = count
    session["last_proactive_at"] = format_time(beijing_now())
    session["last_error"] = ""
    if count >= max(1, min(max_unanswered, MAX_UNANSWERED)):
        session["paused"] = True


def mark_failed(session: dict[str, Any], reason: str) -> None:
    """记录本次主动失败的原因；不改变暂停和未回复计数。"""
    session["last_error"] = str(reason or "")[:300]
