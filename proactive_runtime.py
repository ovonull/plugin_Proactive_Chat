"""主动聊天插件的运行期公共件：配置、存储、后台任务与安全发送。"""

from __future__ import annotations

import asyncio
import re
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from app.services.plugin_sdk import PluginCallError

from proactive_state import (
    ProactiveStateError,
    normalize_state,
)

SCHEDULE_EVENT = "proactive_chat_tick"
DEFAULT_TICK_SECONDS = 30
DEFAULT_SILENCE_MINUTES = 10
DEFAULT_MAX_UNANSWERED = 3
DEFAULT_HISTORY_LIMIT = 12
MAX_STORAGE_UPDATE_ATTEMPTS = 4

# 后台任务代次：源码重载后旧任务作废，避免同一会话重复发送。
_generation: int = 0
_store_lock: asyncio.Lock | None = None
_background_tasks: set[asyncio.Task] = set()


def next_generation() -> int:
    """每次启动巡检时递增，旧回调据此判断自己是否已过期。"""
    global _generation
    _generation += 1
    return _generation


def current_generation() -> int:
    """返回当前代次号。"""
    return _generation


def store_guard() -> asyncio.Lock:
    """返回状态存储读改写锁。"""
    global _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    return _store_lock


def spawn(host: Any, coroutine: Awaitable[Any], operation: str) -> None:
    """启动后台任务并记录未处理异常，避免 10 秒事件超时。"""
    task = asyncio.create_task(coroutine)
    _background_tasks.add(task)

    def done(finished: asyncio.Task) -> None:
        _background_tasks.discard(finished)
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            host.log(
                f"{operation}异常：{type(error).__name__}: {error}",
                "error",
            )

    task.add_done_callback(done)


def config_int(host: Any, key: str, default: int, low: int, high: int) -> int:
    """读取并限制整数配置。"""
    try:
        value = int(host.config.get(key) or 0)
    except (TypeError, ValueError):
        value = default
    if value <= 0:
        value = default
    return max(low, min(value, high))


def config_bool(host: Any, key: str, default: bool) -> bool:
    """读取布尔配置，未配置时用默认值。"""
    raw = host.config.get(key)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def config_text(host: Any, key: str, default: str) -> str:
    """读取文本配置，空值用默认值。"""
    return str(host.config.get(key) or "").strip() or default


def config_float(host: Any, key: str, default: float, low: float, high: float) -> float:
    """读取并限制 0～1 的小数配置（插话概率）。"""
    try:
        value = float(host.config.get(key) if host.config.get(key) not in (None, "") else default)
    except (TypeError, ValueError):
        value = default
    if not 0 <= value <= 1:
        value = default
    return max(low, min(value, high))


def quiet_now(host: Any) -> bool:
    """当前是否落在免打扰时段，格式坏时按不打扰处理。"""
    if not config_bool(host, "quiet_hours_enabled", True):
        return False
    raw = config_text(host, "quiet_hours", "23:00-07:00")
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", raw)
    if match is None:
        return False
    try:
        start_h, start_m, end_h, end_m = (int(part) for part in match.groups())
        if not (0 <= start_h <= 23 and 0 <= end_h <= 23 and start_m < 60 and end_m < 60):
            return False
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone(timedelta(hours=8)))
        minutes = now.hour * 60 + now.minute
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        # 跨零点：23:00-07:00 这类写法；不跨零点时按普通区间。
        return minutes >= start or minutes < end if start > end else start <= minutes < end
    except (TypeError, ValueError):  # pragma: no cover - 上面已校验
        return False


async def register_tick(host: Any) -> None:
    """幂等登记巡检任务，调用方不要持状态锁。"""
    tick = config_int(host, "tick_seconds", DEFAULT_TICK_SECONDS, 10, 3600)
    await host.schedule(
        SCHEDULE_EVENT, tick, {"kind": "proactive_scan"}, preserve_next_run=True
    )


async def load_state(host: Any) -> dict[str, Any]:
    """读状态并校验宿主返回的写校验和。"""
    result = await host.action("storage.read")
    checksum = result.get("checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ProactiveStateError("宿主未返回安全的存储校验和，请先升级 NapCatQQ Service")
    if "data" not in result:
        raise ProactiveStateError("宿主未返回状态数据，本次操作未保存")
    state = normalize_state(result["data"])
    state["_storage_checksum"] = checksum
    return state


async def save_state(host: Any, state: dict[str, Any]) -> None:
    """按读时的校验和条件写入，状态过期时由宿主抛冲突。"""
    await host.action(
        "storage.write",
        {"data": normalize_state(state), "expected_checksum": state["_storage_checksum"]},
    )


async def update_state(
    host: Any, update: Callable[[dict[str, Any]], tuple[bool, Any]]
) -> Any:
    """在最新状态上做同步变更，遇并发冲突有限重试。

    更新回调只操作内存状态，不做网络调用，也不发消息。
    """
    for attempt in range(MAX_STORAGE_UPDATE_ATTEMPTS):
        try:
            async with store_guard():
                state = await load_state(host)
                changed, result = update(state)
                if changed:
                    await save_state(host, state)
                return result
        except PluginCallError as error:
            if not getattr(error, "storage_conflict", False):
                raise
            host.log(
                f"主动聊天状态并发冲突，机器人={host.bot_qq}，重试={attempt + 1}",
                "warning",
            )
            if attempt + 1 >= MAX_STORAGE_UPDATE_ATTEMPTS:
                raise ProactiveStateError("状态正在被频繁更新，本次未保存") from error
            await asyncio.sleep(0)
    raise ProactiveStateError("状态更新未完成")


async def safe_send(host: Any, chat_type: str, target_id: str, text: str) -> bool:
    """按会话类型主动发消息；失败写宿主日志并返回 False。"""
    content = str(text or "").strip()
    if not content:
        return False
    try:
        if chat_type == "group":
            await host.send_group_msg(target_id, content)
        else:
            await host.send_private_msg(target_id, content)
        return True
    except Exception as error:  # noqa: BLE001 - 发送失败只记日志
        host.log(
            f"主动消息发送失败，机器人={host.bot_qq}，会话={chat_type}:{target_id}，"
            f"错误类型={type(error).__name__}，错误={error}\n{traceback.format_exc()}",
            "error",
        )
        return False
