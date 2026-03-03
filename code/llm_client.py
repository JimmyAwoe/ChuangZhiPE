"""
LLM Client — 本文件不允许考生修改。

提供 create_llm_caller() 工厂函数，返回一个 call_llm(messages) 可调用对象。
提供 create_token_counter() 工厂函数，返回一个 count_tokens(text) 可调用对象。
所有模型参数（model, temperature, top_p, max_tokens 等）在此处固定，
考生只能通过 messages 参数与模型交互。
"""

from openai import OpenAI
from transformers import AutoTokenizer

MODEL = "Qwen3-8B"
TOKENIZER_PATH = "Qwen/Qwen3-8B"
MAX_TOKENS = 8192
TEMPERATURE = 1.0
TOP_P = 1
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def create_token_counter():
    """
    返回一个 count_tokens(text) 函数。

    Returns
    -------
    callable
        count_tokens(text: str) -> int
        输入文本，返回 token 数量。
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count_tokens


def create_llm_caller(api_base: str, api_key: str):
    """
    返回一个 call_llm(messages) 函数。

    Parameters
    ----------
    api_base : str
        vLLM / OpenAI-compatible API 的 base URL。
    api_key : str
        API key（本地部署可用 "EMPTY"）。

    Returns
    -------
    callable
        call_llm(messages: list[dict]) -> str
        输入 OpenAI 格式的 messages，返回模型的回复文本。
    """
    client = OpenAI(base_url=api_base, api_key=api_key)

    def call_llm(messages: list[dict]) -> str:
        """
        调用大语言模型。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式的对话消息列表，例如：
            [{"role": "user", "content": "Hello"}]

        Returns
        -------
        str
            模型的回复文本。
        """
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            extra_body=EXTRA_BODY,
        )
        return completion.choices[0].message.content

    return call_llm
