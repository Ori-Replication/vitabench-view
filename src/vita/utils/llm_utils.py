import json
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger
import requests


from vita.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from vita.utils.event_logger import log_event
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


def _safe_endpoint(url: str) -> str:
    """用于日志：只打印 scheme/host/path 前缀，避免泄露 query 中的敏感参数。"""
    try:
        u = urlparse(url)
        path = u.path or ""
        if len(path) > 160:
            path = path[:160] + "…"
        return f"{u.scheme}://{u.netloc}{path}"
    except Exception:
        return "<unparsable-url>"


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add interleaved thinking content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                reasoning_content = message.raw_data["message"].get("reasoning_content")
                if reasoning_content:
                    messages_formatted[-1]["reasoning_content"] = reasoning_content
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            raw_message = original_msg.raw_data.get("message", {})

            # Extract reasoning content and signature
            reasoning_content = raw_message.get("reasoning_content") or raw_message.get("reasoning")
            signature = raw_message.get("signature")
            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content
            if signature:
                messages_formatted[i]["signature"] = signature

    except Exception as e:
        import traceback
        traceback.print_exc()

    return messages_formatted


def to_deepseek_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        for i, (formatted_msg, original_msg) in enumerate(zip(messages_formatted, messages)):
            if formatted_msg.get("role") != "assistant":
                continue

            if not hasattr(original_msg, 'raw_data') or original_msg.raw_data is None:
                continue

            reasoning_content = original_msg.raw_data.get("message", {}).get("reasoning_content", None) or \
                               original_msg.raw_data.get("message", {}).get("reasoning", None)

            if reasoning_content:
                messages_formatted[i]["reasoning_content"] = reasoning_content

    except Exception as e:
        import traceback
        traceback.print_exc()

    return messages_formatted


def kwargs_adapter(data: dict, enable_think: False, messages: list) -> dict:
    if "claude" in data["model"]:
        if not enable_think:
            data["thinking"] = {"type": "disabled"}
        else:
            data["messages"] = to_claude_think_official(data["messages"], messages)
    elif "deepseek" in data["model"]:
        if enable_think:
            data["messages"] = to_deepseek_think_official(data["messages"], messages)
    else:
        if not enable_think:
            if data.get("model", "") == "gpt-5":
                data["reasoning_effort"] = "minimal"
            elif "reasoning_effort" in data:
                data.pop("reasoning_effort")
    return data


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        started_at = time.perf_counter()
        log_event(
            "llm_request_started",
            model=model,
            tool_count=len(tools) if tools else 0,
        )
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        try:
            data = {
                "model": model,
                "messages": messages_formatted,
                "stream": False,
                "temperature": kwargs.get("temperature"),
                "tools": tools,
                "tool_choice": tool_choice,
            }
            data.update(models[model])
            data = kwargs_adapter(data, enable_think, messages)
            headers = models[model]["headers"]

            # 集群网络波动较大：在默认基础上额外增加 3 次重试
            max_retries = 6
            retry_delay = 1
            for attempt in range(max_retries + 1):
                try:
                    http_started = time.perf_counter()
                    logger.info(
                        "[LLM] 请求开始 model={model!r} endpoint={ep} messages={nmsg} tools={ntool} think={think} attempt={att}/{total}",
                        model=model,
                        ep=_safe_endpoint(data["base_url"]),
                        nmsg=len(messages_formatted),
                        ntool=len(tools or []),
                        think=enable_think,
                        att=attempt + 1,
                        total=max_retries + 1,
                    )
                    http_resp = requests.post(
                        data["base_url"], json=data, headers=headers, timeout=(10, 600)
                    )
                    http_elapsed = time.perf_counter() - http_started
                    logger.info(
                        "[LLM] HTTP 返回 status={status} http_elapsed={t:.2f}s total_elapsed={T:.2f}s",
                        status=http_resp.status_code,
                        t=http_elapsed,
                        T=time.perf_counter() - started_at,
                    )
                    if http_resp.status_code >= 400:
                        preview = (http_resp.text or "")[:2000]
                        logger.error(
                            "[LLM] 非 2xx 响应，Content-Type={ct}，正文预览（前 2000 字符）:\n{preview}",
                            ct=http_resp.headers.get("Content-Type", ""),
                            preview=preview,
                        )

                    if http_resp.status_code != 500:
                        try:
                            response = http_resp.json()
                        except json.JSONDecodeError:
                            logger.error(
                                "[LLM] 响应体不是合法 JSON（常见于网关/HTML 错误页）。"
                                "原始前 500 字符:\n{raw}",
                                raw=(http_resp.text or "")[:500],
                            )
                            raise
                        log_event(
                            "llm_request_http_ok",
                            model=model,
                            attempt=attempt + 1,
                            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                        )
                        break

                    if attempt < max_retries:
                        logger.warning(f"API returned 500 error, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        http_resp.raise_for_status()

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        except Exception as e:
            logger.error(e)
            log_event(
                "llm_request_failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            raise e
        usage = get_response_usage(response)
        cost = get_response_cost(usage, model)
        try:
            response = response['choices'][0]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Full response: {json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, dict) else response}")
            raise ValueError(f"Invalid API response format: {e}") from e
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        content = response['message'].get('content')
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=response,
        )
        log_event(
            "llm_request_finished",
            model=model,
            prompt_tokens=usage["prompt_tokens"] if usage else None,
            completion_tokens=usage["completion_tokens"] if usage else None,
            cost=cost,
            tool_call_count=len(tool_calls) if tool_calls is not None else 0,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        logger.info(
            "[LLM] 解析完成 model={model!r} prompt_tokens={pt} completion_tokens={ct} "
            "tool_calls={tc} total_elapsed={T:.2f}s",
            model=model,
            pt=usage["prompt_tokens"] if usage else None,
            ct=usage["completion_tokens"] if usage else None,
            tc=len(tool_calls) if tool_calls is not None else 0,
            T=time.perf_counter() - started_at,
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)
        log_event(
            "llm_request_exception",
            model=model,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost
