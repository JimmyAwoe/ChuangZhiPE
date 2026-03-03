"""
评测运行脚本 — 本文件不允许考生修改（自己调试时可以修改）

用法示例：
    # 运行 dev 集上的第 0 题
    python run.py --question-index 0

    # 运行 dev 集上的所有题目
    python run.py --all

    # 指定最大轮数
    python run.py --all --max-turns 3
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import code_reward
from llm_client import create_llm_caller, create_token_counter


JSONL_PATH = "data/dev.jsonl"
OUTPUT_DIR = "outputs"
API_BASE = "http://localhost:8001/v1"
API_KEY = "EMPTY"
MAX_TURNS = 3
MAX_TEST_CASES = None
SAMPLES = 16


@dataclass
class QuestionData:
    question_id: str
    prompt: str
    tests: Any
    split: str


def _load_from_jsonl(jsonl_path: str, index: int = 0) -> QuestionData:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                row = json.loads(line)
                prompt_content = row["prompt"]
                if isinstance(prompt_content, str):
                    try:
                        prompt_list = json.loads(prompt_content)
                        if isinstance(prompt_list, list) and len(prompt_list) > 0:
                            prompt_content = prompt_list[0].get("content", prompt_content)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(prompt_content, list) and len(prompt_content) > 0:
                    prompt_content = prompt_content[0].get("content", str(prompt_content))

                return QuestionData(
                    question_id=str(row["id"]),
                    prompt=prompt_content,
                    tests=row["tests"],
                    split=row["split"],
                )
    raise IndexError(f"Index {index} out of range in {jsonl_path}")


def count_questions(jsonl_path: str) -> int:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _make_execute_code(question: QuestionData, max_test_cases: Optional[int]):
    """创建 execute_code 闭包，供考生的 solution 调用。"""

    def execute_code(response: str) -> dict:
        """
        提取回复中的代码并在沙箱中运行测试用例。

        Parameters
        ----------
        response : str
            包含 ```python ... ``` 代码块的模型回复文本。

        Returns
        -------
        dict
            {"passed": bool, "feedback": str}
        """
        result = code_reward.compute_score(
            solution=response,
            ground_truth=question.tests,
            extra_info={"split": question.split, "truncated": False},
            max_test_cases=max_test_cases,
            sparse_rewards=True,
        )
        passed = float(result.get("score", 0.0)) >= 1.0
        feedback = result.get("feedback", "")
        return {"passed": passed, "feedback": feedback}

    return execute_code


class _MaxCallsExceeded(Exception):
    pass


def _wrap_with_call_limit(call_llm, max_turns: int):
    """包装 call_llm，强制限制最大调用次数。超出后抛出 _MaxCallsExceeded。"""
    calls = [0]

    def limited_call_llm(messages: list[dict]) -> str:
        calls[0] += 1
        if calls[0] > max_turns:
            raise _MaxCallsExceeded(
                f"已达到最大 LLM 调用次数 ({max_turns})，不允许继续调用。"
            )
        return call_llm(messages)

    return limited_call_llm


def run_single_question(
    question_index: int,
    jsonl_path: str = JSONL_PATH,
    run_name: str = None,
    api_base: str = API_BASE,
    api_key: str = API_KEY,
    max_turns: int = MAX_TURNS,
    max_test_cases: Optional[int] = MAX_TEST_CASES,
    output_dir: str = OUTPUT_DIR,
    sample_id: int = 0,
) -> Dict[str, Any]:
    """运行单个题目的评测（一个 sample）。"""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    question = _load_from_jsonl(jsonl_path, question_index)
    raw_call_llm = create_llm_caller(api_base, api_key)
    call_llm = _wrap_with_call_limit(raw_call_llm, max_turns)
    count_tokens = create_token_counter()
    execute_code = _make_execute_code(question, max_test_cases)

    from solution import run_question

    t_start = time.time()
    try:
        final_response = run_question(
            question_prompt=question.prompt,
            call_llm=call_llm,
            execute_code=execute_code,
            max_turns=max_turns,
            count_tokens=count_tokens,
        )
    except _MaxCallsExceeded:
        final_response = ""
    elapsed = time.time() - t_start

    final_result = code_reward.compute_score(
        solution=final_response,
        ground_truth=question.tests,
        extra_info={"split": question.split, "truncated": False},
        max_test_cases=None,
        sparse_rewards=True,
    )
    final_reward = float(final_result.get("score", 0.0))

    out_dir = Path(output_dir)
    if run_name:
        out_dir = out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "run_name": run_name,
        "question_index": question_index,
        "question_id": question.question_id,
        "split": question.split,
        "sample_id": sample_id,
        "final_reward": final_reward,
        "passed": final_reward >= 1.0,
        "elapsed_seconds": elapsed,
        "response": final_response,
    }

    record_path = out_dir / f"q{question_index}-s{sample_id}.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return {
        "question_index": question_index,
        "question_id": question.question_id,
        "split": question.split,
        "sample_id": sample_id,
        "final_reward": final_reward,
        "passed": final_reward >= 1.0,
        "elapsed_seconds": elapsed,
    }


def _run_one(kwargs):
    """ProcessPoolExecutor wrapper."""
    try:
        return run_single_question(**kwargs)
    except Exception as e:
        return {
            "question_index": kwargs["question_index"],
            "question_id": "unknown",
            "split": "unknown",
            "sample_id": kwargs.get("sample_id", 0),
            "final_reward": 0.0,
            "passed": False,
            "elapsed_seconds": 0.0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Prompt Engineering Exam Runner")
    parser.add_argument("--jsonl-path", type=str, default=JSONL_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=API_BASE)
    parser.add_argument("--api-key", type=str, default=API_KEY)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples per question")
    parser.add_argument("--workers", type=int, default=80, help="Number of parallel workers")
    parser.add_argument("--question-index", type=int, default=None, help="Run a single question by index")
    parser.add_argument("--all", action="store_true", help="Run all questions")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.question_index is not None and not args.all:
        result = run_single_question(
            question_index=args.question_index,
            jsonl_path=args.jsonl_path,
            run_name=args.run_name,
            api_base=args.api_base,
            api_key=args.api_key,
            max_turns=args.max_turns,
            max_test_cases=args.max_test_cases,
            output_dir=args.output_dir,
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Q{result['question_index']} ({result['question_id']}) "
              f"reward={result['final_reward']:.2f} {status} "
              f"({result['elapsed_seconds']:.1f}s)")
        return

    if not args.all:
        parser.error("Please specify --question-index or --all")

    total = count_questions(args.jsonl_path)
    indices = list(range(total))
    num_tasks = total * args.samples

    if args.run_name is None:
        args.run_name = f"run_{int(time.time())}"

    print(f"Questions: {total}, samples/question: {args.samples}, "
          f"total tasks: {num_tasks}, workers: {args.workers}, "
          f"max_turns: {args.max_turns}")

    task_kwargs = [
        {
            "question_index": idx,
            "sample_id": sid,
            "jsonl_path": args.jsonl_path,
            "run_name": args.run_name,
            "api_base": args.api_base,
            "api_key": args.api_key,
            "max_turns": args.max_turns,
            "max_test_cases": args.max_test_cases,
            "output_dir": args.output_dir,
        }
        for idx in indices
        for sid in range(args.samples)
    ]

    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, kw): (kw["question_index"], kw["sample_id"])
            for kw in task_kwargs
        }
        for future in as_completed(futures):
            idx, sid = futures[future]
            result = future.result()
            results.append(result)
            status = "PASS" if result.get("passed") else "FAIL"
            err = f" [{result['error']}]" if "error" in result else ""
            print(
                f"[{len(results):>4}/{num_tasks}] Q{idx:>3} s{sid}  "
                f"({result.get('question_id', '?'):>6}) "
                f"reward={result.get('final_reward', 0):.2f}  "
                f"{status}{err}"
            )

    elapsed = time.time() - t_start

    by_question = defaultdict(list)
    for r in results:
        by_question[r["question_index"]].append(r)

    question_pass_at_1 = {}
    for q_idx in sorted(by_question.keys()):
        samples = by_question[q_idx]
        n_passed = sum(1 for s in samples if s.get("passed"))
        question_pass_at_1[q_idx] = n_passed / len(samples)

    avg_pass_at_1 = (sum(question_pass_at_1.values()) / len(question_pass_at_1)
                     if question_pass_at_1 else 0.0)

    total_samples = len(results)
    total_passed = sum(1 for r in results if r.get("passed"))
    total_errors = sum(1 for r in results if "error" in r)

    print("\n" + "=" * 60)
    print(f"Evaluation complete in {elapsed:.1f}s")
    print(f"  Questions:          {len(question_pass_at_1)}")
    print(f"  Samples/question:   {args.samples}")
    print(f"  Max turns:          {args.max_turns}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Total passed:       {total_passed}")
    print(f"  Total errors:       {total_errors}")
    print(f"  Avg pass@1:         {avg_pass_at_1:.4f}")
    print()

    for q_idx in sorted(question_pass_at_1.keys()):
        samples = by_question[q_idx]
        qid = samples[0].get("question_id", "?")
        p1 = question_pass_at_1[q_idx]
        print(f"  Q{q_idx:>3} ({qid:>6})  pass@1={p1:.2f}")

    print()
    print(f"  === Average pass@1: {avg_pass_at_1:.4f} ===")
    print("=" * 60)

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary.json"

    per_question_summary = []
    for q_idx in sorted(question_pass_at_1.keys()):
        samples = by_question[q_idx]
        per_question_summary.append({
            "question_index": q_idx,
            "question_id": samples[0].get("question_id", "?"),
            "pass_at_1": question_pass_at_1[q_idx],
            "samples": sorted(samples, key=lambda s: s["sample_id"]),
        })

    summary = {
        "run_name": args.run_name,
        "jsonl_path": args.jsonl_path,
        "max_turns": args.max_turns,
        "num_questions": len(question_pass_at_1),
        "samples_per_question": args.samples,
        "total_samples": total_samples,
        "total_passed": total_passed,
        "total_errors": total_errors,
        "avg_pass_at_1": avg_pass_at_1,
        "elapsed_seconds": elapsed,
        "per_question": per_question_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
