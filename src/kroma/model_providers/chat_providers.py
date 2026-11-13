import os
import time
import random
import re
from typing import Any, Dict, List, Tuple
from openai import RateLimitError, APITimeoutError, APIConnectionError

from .providers import ChatProvider, parse_chat_response


class OpenAIProvider(ChatProvider):
    """
    ChatProvider implementation for OpenAI-compatible APIs (OpenAI, Groq, SiliconFlow, etc.)
    with automatic provider routing and advanced rate limit & timeout handling.
    """

    def __init__(self, api_key: str = None):
        from openai import OpenAI

        # 1. 尝试从环境变量读取各平台专用配置
        groq_key = os.environ.get("GROQ_API_KEY")
        groq_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        silicon_key = os.environ.get("SILICONFLOW_API_KEY")
        silicon_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

        default_key = os.environ.get("OPENAI_API_KEY") or api_key
        default_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # 2. 依据 Key 前缀或 URL 特征自动判定目标平台
        target_key = default_key
        target_url = default_url

        if target_key:
            if target_key.startswith("gsk_"):
                # Groq Key 自动绑定 Groq 端点
                target_url = groq_url
            elif target_key.startswith("sk-") and ("silicon" in str(silicon_url) or "silicon" in str(default_url)):
                # SiliconFlow Key 绑定硅基流动端点
                target_url = silicon_url

        # 初始化客户端（加入 30 秒超时限制）
        self.client = OpenAI(
            api_key=target_key,
            base_url=target_url,
            timeout=30.0,
            max_retries=2
        )

    def create_completion(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float,
            max_tokens: int,
            response_format: Any = None,
    ) -> Any:
        max_retries = 5
        base_delay = 3.0

        for attempt in range(max_retries):
            try:
                # 执行请求
                if response_format:
                    try:
                        resp = self.client.beta.chat.completions.parse(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                    except Exception:
                        resp = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                else:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                # 每次成功请求后主动休眠 1.5 秒，平滑 RPM 避免触发免费层突发频次限制
                time.sleep(1.5)
                return resp

            except RateLimitError as e:
                err_msg = str(e)

                # 判断是否为每日额度(TPD)耗尽
                if "tokens per day" in err_msg.lower() or "tpd" in err_msg.lower():
                    print(f"\n❌ [FATAL] 模型 {model} 的每日 Token 额度 (TPD) 已耗尽！")
                    print("💡 解决方案：请切换为其他模型、更换账号 API Key，或等待额度重置。")
                    raise RuntimeError(f"每日额度耗尽: {model}") from e

                # 若达到最大重试次数则放弃
                if attempt == max_retries - 1:
                    raise RuntimeError(f"连续 {max_retries} 次触发 RPM 限流，放弃重试。") from e

                # 提取错误信息中的建议等待时长
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', err_msg)
                if match:
                    minutes = int(match.group(1)) if match.group(1) else 0
                    seconds = float(match.group(2))
                    wait_time = minutes * 60 + seconds + 1

                print(f"\n⚠️ [WARNING] 触发 API 速率限制 (RPM)，等待 {wait_time:.2f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                time.sleep(wait_time)

            except (APITimeoutError, APIConnectionError) as e:
                # 捕获网络超时或连接中断，自动退避重试，避免永久卡死
                if attempt == max_retries - 1:
                    raise RuntimeError(f"连续 {max_retries} 次网络连接或请求超时，放弃重试。") from e

                wait_time = base_delay * (2 ** attempt) + random.uniform(1, 2)
                print(f"\n⚠️ [WARNING] 请求超时或连接异常 ({type(e).__name__})，等待 {wait_time:.2f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                time.sleep(wait_time)

            except Exception as e:
                # 其他异常直接抛出
                raise e

        raise RuntimeError("重试次数耗尽，API 仍不可用。")

    def parse_response(
            self,
            response: Any,
            include_tokens: bool
    ) -> Tuple[str, int, int]:
        return parse_chat_response(response, include_tokens)


class TogetherProvider(ChatProvider):
    """
    ChatProvider implementation for TogetherAI.
    """

    def __init__(self, api_key: str):
        from together import Together
        self.client = Together(api_key=api_key)

    def create_completion(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float,
            max_tokens: int,
            response_format: Any = None,
    ) -> Any:
        if response_format:
            return self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def parse_response(
            self,
            response: Any,
            include_tokens: bool
    ) -> Tuple[str, int, int]:
        return parse_chat_response(response, include_tokens)