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

    # ────────────────────── 工具函数 ──────────────────────

    def trim_to_token_budget(text: str, max_tokens: int) -> str:
        if not text:
            return ""
        if count_tokens(text) <= max_tokens:
            return text
        left, right = 0, len(text)
        best = ""
        while left <= right:
            mid = (left + right) // 2
            candidate = text[:mid]
            if count_tokens(candidate) <= max_tokens:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1
        return best

    def extract_python_blocks(text: str) -> list:
        if not text:
            return []
        python_blocks, generic_blocks = [], []
        i = 0
        n = len(text)
        while i < n:
            start = text.find("```", i)
            if start == -1:
                break
            line_end = text.find("\n", start + 3)
            if line_end == -1:
                break
            fence_info = text[start + 3:line_end].strip().lower()
            end = text.find("```", line_end + 1)
            if end == -1:
                break
            code = text[line_end + 1:end].strip()
            if code:
                if "python" in fence_info:
                    python_blocks.append(code)
                elif not fence_info:
                    generic_blocks.append(code)
            i = end + 3
        blocks = python_blocks if python_blocks else generic_blocks
        if not blocks:
            return []
        seen, dedup = set(), []
        for b in blocks:
            if b not in seen:
                seen.add(b)
                dedup.append(b)
        return dedup

    def wrap_code(code: str) -> str:
        return "```python\n" + (code or "").strip() + "\n```"

    def normalize_feedback(feedback: str, limit_chars: int = 2400) -> str:
        if not feedback:
            return "No detailed feedback."
        feedback = feedback.strip()
        if len(feedback) <= limit_chars:
            return feedback
        keep_head = int(limit_chars * 0.65)
        keep_tail = limit_chars - keep_head - 20
        return feedback[:keep_head] + "\n...(truncated)...\n" + feedback[-keep_tail:]

    def detect_error_type(feedback: str) -> str:
        """根据反馈文本判断错误类型，返回针对性修复提示。"""
        fb = (feedback or "").lower()

        tle_hint = (
            "The error is TIME LIMIT EXCEEDED (TLE). Your algorithm is too slow.\n"
            "Fix by choosing a faster algorithm:\n"
            "- Replace O(N²) loops with binary search, two pointers, or hash maps\n"
            "- Use BFS/DFS instead of recursive simulation\n"
            "- Use segment trees, Fenwick trees, or sorted containers for range queries\n"
            "- Precompute prefix sums/products instead of recomputing\n"
            "- For string problems: use KMP, Z-function, or rolling hash instead of naive matching\n"
            "- Use sys.stdin.read() and split() for bulk input parsing"
        )
        wa_hint = (
            "The error is WRONG ANSWER (WA). Your logic has a bug.\n"
            "Debug by checking:\n"
            "- Edge cases: N=0, N=1, all equal values, already sorted input, empty string\n"
            "- Off-by-one errors in loops and indices\n"
            "- Integer overflow (Python is fine, but check logic)\n"
            "- Output format: extra spaces, wrong newlines, 0-indexed vs 1-indexed\n"
            "- Re-read the problem: check if you misunderstood a constraint or formula\n"
            "- Modular arithmetic: ensure all intermediate computations are taken mod p"
        )
        re_hint = (
            "The error is a RUNTIME ERROR. Your code crashes.\n"
            "Fix by checking:\n"
            "- Array index out of bounds\n"
            "- Division by zero\n"
            "- Input parsing: use int() / split() correctly\n"
            "- Recursion depth: increase sys.setrecursionlimit if needed, or convert to iterative\n"
            "- Empty list/string access"
        )
        syntax_hint = (
            "The error is a SYNTAX ERROR or incorrect code format.\n"
            "Ensure your code:\n"
            "- Has correct Python indentation\n"
            "- Has no missing colons or parentheses\n"
            "- Is enclosed in a single ```python code block"
        )

        if any(k in fb for k in ["time limit exceeded", "tle", "time limit", "timed out"]):
            return tle_hint
        if any(k in fb for k in ["wrong answer", "wrong output", "expected", "mismatch"]):
            return wa_hint
        if any(k in fb for k in ["runtime error", "traceback", "error:", "exception", "segfault"]):
            return re_hint
        if any(k in fb for k in ["syntaxerror", "syntax error", "indentationerror", "invalid syntax"]):
            return syntax_hint
        # default — give WA hint as it's most common
        return wa_hint

    def evaluate_response(response_text: str) -> tuple:
        """评估LLM回复中的所有代码候选，返回(best_response, passed, best_feedback)。"""
        blocks = extract_python_blocks(response_text)
        candidates = []
        if blocks:
            for code in blocks[:6]:
                candidates.append(wrap_code(code))
        elif response_text and response_text.strip():
            candidates.append(response_text)

        if not candidates:
            return "", False, "Model returned empty output."

        best_response = candidates[0]
        best_feedback = "Unknown error."
        best_priority = -1

        def _priority(fb: str) -> int:
            fb_low = (fb or "").lower()
            if any(k in fb_low for k in ["time limit exceeded", "tle"]):
                return 4
            if any(k in fb_low for k in ["wrong answer", "expected", "mismatch"]):
                return 3
            if any(k in fb_low for k in ["runtime error", "traceback"]):
                return 2
            if any(k in fb_low for k in ["syntaxerror", "syntax error"]):
                return 0
            return 1

        for candidate in candidates:
            result = execute_code(candidate)
            if result.get("passed"):
                return candidate, True, ""
            feedback = normalize_feedback(result.get("feedback", ""))
            prio = _priority(feedback)
            if prio > best_priority or best_response == candidates[0]:
                best_priority = prio
                best_feedback = feedback
                best_response = candidate

        return best_response, False, best_feedback

    # ────────────────────── 提示词定义 ──────────────────────

    system_prompt = (
        "You are an expert competitive programmer.\n"
        "Key guidelines:\n"
        "- Always use `import sys; input = sys.stdin.readline` (or sys.stdin.read()) for fast I/O\n"
        "- Choose algorithm by constraint size: N≤10^3 → O(N²) ok; N≤10^5 → O(N log N) needed; "
        "N≤10^6 → O(N) needed\n"
        "- CRITICAL: Re-read the output specification carefully. The output index may differ from "
        "the input index — build a reverse mapping if needed (e.g. inverse permutation)\n"
        "- Output complete, runnable Python in ```python code blocks"
    )

    first_user_prompt = (
        question_prompt
        + "\n\n"
        "Provide TWO complete Python solutions as separate ```python code blocks.\n"
        "Solution 1: Your primary solution with the optimal algorithm.\n"
        "Solution 2: An alternative solution using a DIFFERENT algorithm or approach "
        "(different data structure, different strategy, or simpler brute-force if constraints allow).\n\n"
        "Requirements for EACH solution:\n"
        "- First line: `import sys; input = sys.stdin.readline` for line-by-line input, "
        "OR `data = sys.stdin.read().split(); idx = 0` if you prefer reading all input at once\n"
        "- NEVER use a single `input().split()` call to read multi-line input\n"
        "- Add a comment: `# Algorithm: <name>, Time: O(<complexity>)`\n"
        "- BEFORE CODING: carefully identify what the OUTPUT index represents "
        "(it may differ from the input index — build reverse mappings if needed)\n"
        "- After writing the solution, mentally verify it produces the correct output "
        "for Sample Input 1\n"
        "- Handle all edge cases from the constraints\n"
        "- Ensure each solution is complete and executable"
    )

    TOKEN_BUDGET = 14500  # 单次调用最大 token 数（留余量）

    # ────────────────────── 主循环 ──────────────────────

    best_response = ""
    best_feedback = ""
    turn1_full_response = ""  # 保留第一轮完整回复用于多轮对话

    for turn in range(min(max_turns, 3)):

        if turn == 0:
            # — Turn 1: 首次求解 —
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_user_prompt},
            ]

        elif turn == 1:
            # — Turn 2: 基于反馈修复，优先使用多轮对话保留上下文 —
            error_hint = detect_error_type(best_feedback)

            repair_followup = (
                "Your solution(s) failed the hidden tests.\n\n"
                "Execution feedback:\n"
                + best_feedback
                + "\n\n"
                + error_hint
                + "\n\n"
                "Provide TWO corrected Python solutions as separate ```python code blocks.\n"
                "Each solution must fix the identified issue and handle all edge cases."
            )

            # 尝试多轮对话（携带上下文）
            candidate_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_user_prompt},
                {"role": "assistant", "content": turn1_full_response},
                {"role": "user", "content": repair_followup},
            ]
            total_tokens = sum(count_tokens(m["content"]) for m in candidate_messages)

            if total_tokens <= TOKEN_BUDGET:
                messages = candidate_messages
            else:
                # token 超预算，回退到新鲜 prompt
                short_q = trim_to_token_budget(question_prompt, 6000)
                short_code = trim_to_token_budget(best_response, 2500)
                short_fb = trim_to_token_budget(best_feedback, 1500)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Fix this failing Python solution.\n\n"
                            "Problem:\n" + short_q + "\n\n"
                            "Failed code:\n" + short_code + "\n\n"
                            "Execution feedback:\n" + short_fb + "\n\n"
                            + detect_error_type(best_feedback) + "\n\n"
                            "Provide TWO corrected ```python code blocks."
                        ),
                    },
                ]

        else:
            # — Turn 3: 最终修复，新鲜 prompt，聚焦关键错误 —
            error_hint = detect_error_type(best_feedback)
            short_q = trim_to_token_budget(question_prompt, 7000)
            short_code = trim_to_token_budget(best_response, 2000)
            short_fb = trim_to_token_budget(best_feedback, 1200)

            final_repair_prompt = (
                "This is your last attempt. Fix the solution carefully.\n\n"
                "Problem:\n" + short_q + "\n\n"
                "Previous best code:\n" + short_code + "\n\n"
                "Execution feedback:\n" + short_fb + "\n\n"
                + error_hint + "\n\n"
                "Provide TWO corrected ```python code blocks. "
                "If you suspect a fundamentally wrong algorithm, try a completely different approach."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_repair_prompt},
            ]

        response = call_llm(messages)

        if turn == 0:
            turn1_full_response = response

        candidate_response, passed, feedback = evaluate_response(response)

        if candidate_response:
            best_response = candidate_response
        if feedback:
            best_feedback = feedback

        if passed:
            return candidate_response

    return best_response or "```python\n# fallback\nprint('')\n```"
