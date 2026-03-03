"""
考生答题文件 — 这是你唯一需要编辑的文件。

你需要实现 run_question 函数，利用提供的 call_llm, execute_code和count_tokens 工具，
尽可能正确地解决给定的编程题目。

可用工具说明
-----------
call_llm(messages: list[dict]) -> str
    调用大语言模型。输入 OpenAI 格式的 messages 列表，返回模型回复文本。
    示例：
        response = call_llm([{"role": "user", "content": "Hello"}])

execute_code(response: str) -> dict
    提取回复中的代码并在沙箱中运行测试。输入为包含 ```python ... ``` 代码块的文本。
    你可以直接传入 call_llm 的返回值。
    返回一个字典，包含：
        - "passed": bool — 是否通过所有测试用例
        - "feedback": str — 如果未通过，包含错误信息/测试反馈；如果通过则为空字符串

count_tokens(text: str) -> int
    计算文本的 token 数量。可用于管理上下文长度，避免超出模型限制。
    示例：
        n = count_tokens("Hello, world!")

规则
----
1. 你只能通过 call_llm 的 messages 参数与模型交互，不能修改模型参数。
2. 你可以进行多轮调用（调用 call_llm → 提取代码 → execute_code 获取反馈 → 再次调用 call_llm ...）。
3. 最大调用轮数为 max_turns（由评测系统传入），请合理利用。
4. 函数最终需要返回你认为最佳的代码回复（包含 ```code``` 代码块的完整回复文本）。
"""


def run_question(
    question_prompt: str,
    call_llm,
    execute_code,
    max_turns: int,
    count_tokens,
) -> str:
    """
    解决一道编程题。

    Parameters
    ----------
    question_prompt : str
        题目描述（完整的 prompt 文本）。
    call_llm : callable
        调用大语言模型的函数，签名为 call_llm(messages: list[dict]) -> str。
    execute_code : callable
        代码执行与测试函数，签名为 execute_code(response: str) -> dict。
        输入包含 ```python ... ``` 代码块的文本，返回 {"passed": bool, "feedback": str}。
    max_turns : int
        最大允许的 LLM 调用轮数。
    count_tokens : callable
        Token 计数函数，签名为 count_tokens(text: str) -> int。

    Returns
    -------
    str
        最终的模型回复文本（应包含 ```python ... ``` 代码块）。
    """

    # ============================================================
    # TODO: 在下方实现你的 prompt engineering 策略
    # 下面是一个最简单的 baseline 示例（单轮调用，不使用 code interpreter）
    # 你也可以修改为多轮策略以获得更高分数，但注意 max_turns 的限制
    # ============================================================

    messages = [{"role": "user", "content": question_prompt}]
    response = call_llm(messages)

    return response
