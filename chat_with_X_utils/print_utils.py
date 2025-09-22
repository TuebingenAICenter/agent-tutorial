"""
Printing utilities for LangGraph streaming and state outputs.
"""
from typing import Set, Dict, Any
from langchain_core.messages import ToolMessage
import json

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    Console = None
    Panel = None

console = Console() if Console else None


def _format_message_content_rich_like(message) -> str:
    """Format a LangChain message similar to notebooks/utils.py."""
    parts = []
    tool_calls_processed = False

    # Main content
    content = getattr(message, "content", "")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        # Anthropic-style content with tool_use items
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict) and item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item.get('name')}")
                parts.append(f"   Args: {json.dumps(item.get('input', {}), indent=2, ensure_ascii=False)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(content))

    # OpenAI-style tool_calls attached to the message
    if not tool_calls_processed and hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            # tool_calls can be pydantic-like or dict-like; handle both
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("function", {}).get("name")
                args = tc.get("args") if "args" in tc else tc.get("function", {}).get("arguments")
                call_id = tc.get("id")
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", None)
                call_id = getattr(tc, "id", None)

            parts.append(f"\n🔧 Tool Call: {name or 'tool'}")
            if args is not None:
                # Try to load stringified JSON for readability
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                parts.append("   Args: " + json.dumps(args, indent=2, ensure_ascii=False))
            if call_id:
                parts.append(f"   ID: {call_id}")

    # Helpful linkage when debugging tool messages
    if message.__class__.__name__.endswith("ToolMessage") and hasattr(message, "tool_call_id"):
        parts.append(f"\nCall ID: {getattr(message, 'tool_call_id')}")

    return "\n".join([p for p in parts if p])


def _panel_title_and_style_for_message(message):
    cls = message.__class__.__name__
    label = cls.replace("Message", "")
    lower = label.lower()
    if lower.startswith("human"):
        return "👤 Human", "blue"
    if lower.startswith("ai"):
        return "🤖 Assistant", "green"
    if "tool" in lower:
        return "🔧 Tool Output", "yellow"
    return f"📝 {label}", "white"


def _print_one_message_pretty(message, max_length: int):
    # If rich isn't available, fall back to your previous repr
    if console is None or Panel is None:
        msg_repr = getattr(message, "pretty_repr", lambda **_: str(message))(html=True)
        if len(msg_repr) > max_length:
            msg_repr = msg_repr[:max_length] + " ... (truncated)"
        print(msg_repr)
        # Preserve original extra linkage
        if message.__class__.__name__.endswith("ToolMessage") and hasattr(message, "tool_call_id"):
            print(f"Call ID: {message.tool_call_id}\n")
        return

    content = _format_message_content_rich_like(message)
    if max_length and len(content) > max_length:
        content = content[:max_length] + " ... (truncated)"
    title, style = _panel_title_and_style_for_message(message)
    console.print(Panel(content, title=title, border_style=style))


def print_messages_from_stream_event(event: Dict[str, Any], _printed: Set[str], max_length: int = 1500, messages_key: str = "messages") -> None:
    messages = event.get(messages_key)
    if not messages:
        return
    if not isinstance(messages, list):
        messages = [messages]
    for message in messages:
        msg_id = getattr(message, "id", None)
        if msg_id is not None and str(msg_id) in _printed:
            continue
        _print_one_message_pretty(message, max_length)
        if msg_id is not None:
            _printed.add(str(msg_id))


def print_messages_from_state(state: Dict[str, Any], _printed: Set[str], max_length: int = 1500, messages_key: str = "messages") -> None:
    messages = state.get(messages_key, [])
    if not messages:
        return
    for message in messages:
        msg_id = getattr(message, "id", None)
        if msg_id is not None and str(msg_id) in _printed:
            continue
        _print_one_message_pretty(message, max_length)
        if msg_id is not None:
            _printed.add(str(msg_id))