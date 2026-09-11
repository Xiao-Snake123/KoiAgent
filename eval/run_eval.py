"""KoiAgent 评估 Harness（可离线运行，无需任何 API）。

用法::

    python eval/run_eval.py                  # 离线评估：注入防护 + 规则路由 + RAG 检索
    python eval/run_eval.py --with-llm       # 追加评估 LLM 路由兜底（需要 API_KEY）

评估项
------
1. **输入防护**：Prompt 注入**召回率**与**误伤率（FPR）**。
2. **意图路由**：规则路由基线准确率；``--with-llm`` 时评估「规则 + LLM 兜底」完整准确率。
3. **RAG 检索**：Top-K 命中率与关键词覆盖率。

退出码
------
未达 ``--min-*`` / 超出 ``--max-*`` 阈值时返回 1，可直接作为 CI 质量门禁。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(EVAL_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 1. 输入侧 Prompt 注入防护
# --------------------------------------------------------------------------- #
def eval_guard() -> Dict[str, Any]:
    from koiagent.agent.guard import screen

    cases = _load("guard_cases.json")["cases"]
    rows: List[Dict[str, Any]] = []
    tp = fp = tn = fn = 0

    for case in cases:
        _, verdict = screen(case["message"])
        predicted = "block" if verdict.blocked else "allow"
        expect = case["expect"]

        if expect == "block":
            tp += int(predicted == "block")
            fn += int(predicted != "block")
        else:
            tn += int(predicted == "allow")
            fp += int(predicted != "allow")

        rows.append(
            {
                "message": case["message"],
                "expect": expect,
                "predicted": predicted,
                "score": verdict.score,
                "reasons": verdict.reasons,
                "ok": predicted == expect,
            }
        )

    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (len(cases) or 1)
    return {
        "rows": rows,
        "total": len(cases),
        "recall": recall,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# --------------------------------------------------------------------------- #
# 2. 意图路由
# --------------------------------------------------------------------------- #
def eval_intent(with_llm: bool) -> Dict[str, Any]:
    from koiagent.agent.graph import KoiReplyBot

    cases: List[Dict[str, str]] = _load("intent_cases.json")["cases"]

    bot = None
    if with_llm:
        try:
            bot = KoiReplyBot()
        except Exception as e:  # 缺少可用 API 时自动降级为纯规则评估
            print(f"[warn] LLM 路由不可用（{e}），仅评估规则路由")

    rows: List[Dict[str, Any]] = []
    rule_correct = 0
    full_correct = 0
    llm_used = 0

    for case in cases:
        message, expect = case["message"], case["expect"]
        rule_pred = KoiReplyBot._rule_based_intent(message)

        if rule_pred is not None:
            final_pred, source = rule_pred, "rule"
        elif bot is not None:
            final_pred = asyncio.run(
                bot._classify({"user_msg": message, "item_desc": ""})
            ).get("intent", "default")
            source = "llm"
            llm_used += 1
        else:
            final_pred, source = "default", "fallback"

        rule_correct += int((rule_pred or "default") == expect)
        full_correct += int(final_pred == expect)
        rows.append(
            {
                "message": message,
                "expect": expect,
                "rule": rule_pred or "-",
                "final": final_pred,
                "source": source,
                "ok": final_pred == expect,
            }
        )

    total = len(cases) or 1
    return {
        "rows": rows,
        "total": len(cases),
        "rule_only_accuracy": rule_correct / total,
        "full_accuracy": full_correct / total,
        "llm_used": llm_used,
        "with_llm": bot is not None,
    }


# --------------------------------------------------------------------------- #
# 3. RAG 检索
# --------------------------------------------------------------------------- #
def eval_retrieval(top_k: int) -> Dict[str, Any]:
    from koiagent.rag.knowledge import KnowledgeBase

    cases = _load("retrieval_cases.json")["cases"]
    kb = KnowledgeBase()

    rows: List[Dict[str, Any]] = []
    hits = 0
    coverage = 0.0

    for case in cases:
        results = kb.search(case["query"], k=top_k)
        texts = [text for _, text in results]
        keywords = case.get("expect_keywords", [])
        found = [kw for kw in keywords if any(kw in text for text in texts)]

        source_ok = True
        if case.get("expect_source"):
            source_ok = any(case["expect_source"] in src for src, _ in results)
        ok = source_ok and len(found) == len(keywords)

        hits += int(ok)
        coverage += (len(found) / len(keywords)) if keywords else 1.0
        rows.append(
            {
                "query": case["query"],
                "expect_keywords": keywords,
                "found": found,
                "source": results[0][0] if results else "-",
                "ok": ok,
            }
        )

    total = len(cases) or 1
    return {
        "rows": rows,
        "total": len(cases),
        "top_k": top_k,
        "mode": kb.mode,
        "hit_rate": hits / total,
        "keyword_coverage": coverage / total,
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _print_guard(result: Dict[str, Any]) -> None:
    print("\n=== 输入侧 Prompt 注入防护评估 ===")
    for row in result["rows"]:
        flag = "PASS" if row["ok"] else "FAIL"
        reasons = "/".join(row["reasons"]) or "-"
        print(
            f"  [{flag}] 期望={row['expect']:<5} 实测={row['predicted']:<5} "
            f"score={row['score']:<3} {reasons:<16} <- {row['message'][:32]}"
        )
    print(f"  注入召回率     : {result['recall']:.1%}  (TP={result['tp']} FN={result['fn']})")
    print(f"  误伤率 FPR     : {result['false_positive_rate']:.1%}  (FP={result['fp']} TN={result['tn']})")
    print(f"  整体准确率     : {result['accuracy']:.1%}")


def _print_intent(result: Dict[str, Any]) -> None:
    print("\n=== 意图路由评估 ===")
    for row in result["rows"]:
        flag = "PASS" if row["ok"] else "FAIL"
        print(
            f"  [{flag}] {row['message']:<14} 期望={row['expect']:<9} "
            f"规则={row['rule']:<9} 最终={row['final']:<9} 来源={row['source']}"
        )
    print(f"  规则路由准确率 : {result['rule_only_accuracy']:.1%}")
    print(f"  完整准确率     : {result['full_accuracy']:.1%}（LLM 兜底 {result['llm_used']} 次）")


def _print_retrieval(result: Dict[str, Any]) -> None:
    print("\n=== RAG 检索评估 ===")
    print(f"  检索模式: {result['mode']} | Top-K: {result['top_k']}")
    for row in result["rows"]:
        flag = "PASS" if row["ok"] else "FAIL"
        print(
            f"  [{flag}] {row['query']:<16} 期望关键词={'/'.join(row['expect_keywords']):<14} "
            f"命中={'/'.join(row['found']) or '-'}"
        )
    print(f"  命中率         : {result['hit_rate']:.1%}")
    print(f"  关键词覆盖率   : {result['keyword_coverage']:.1%}")


def _write_report(
    path: str,
    guard: Dict[str, Any],
    intent: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> None:
    lines = [
        "# KoiAgent 评估报告",
        "",
        "## 汇总",
        "",
        f"- 注入防护召回率：**{guard['recall']:.1%}**（FP={guard['fp']} FN={guard['fn']}）",
        f"- 注入防护误伤率 FPR：**{guard['false_positive_rate']:.1%}**",
        f"- 意图路由（规则基线）准确率：**{intent['rule_only_accuracy']:.1%}**",
        f"- 意图路由（完整）准确率：**{intent['full_accuracy']:.1%}**"
        + ("（含 LLM 兜底）" if intent["with_llm"] else "（未启用 LLM）"),
        f"- RAG 检索 Top-{retrieval['top_k']} 命中率：**{retrieval['hit_rate']:.1%}**",
        f"- 关键词覆盖率：**{retrieval['keyword_coverage']:.1%}**",
        f"- 检索模式：`{retrieval['mode']}`",
        "",
        "## 注入防护明细",
        "",
        "| 输入 | 期望 | 实测 | 风险分 | 命中规则 | 结果 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in guard["rows"]:
        lines.append(
            f"| {row['message']} | {row['expect']} | {row['predicted']} | {row['score']} | "
            f"{'/'.join(row['reasons']) or '-'} | {'PASS' if row['ok'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## 意图路由明细",
        "",
        "| 消息 | 期望 | 规则 | 最终 | 来源 | 结果 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in intent["rows"]:
        lines.append(
            f"| {row['message']} | {row['expect']} | {row['rule']} | {row['final']} | "
            f"{row['source']} | {'PASS' if row['ok'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## RAG 检索明细",
        "",
        "| 查询 | 期望关键词 | 命中关键词 | 来源 | 结果 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in retrieval["rows"]:
        lines.append(
            f"| {row['query']} | {'/'.join(row['expect_keywords'])} | "
            f"{'/'.join(row['found']) or '-'} | {row['source']} | {'PASS' if row['ok'] else 'FAIL'} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="KoiAgent 评估 Harness")
    parser.add_argument("--with-llm", action="store_true", help="额外评估 LLM 路由兜底（需 API_KEY）")
    parser.add_argument("--top-k", type=int, default=1, help="检索评估的 Top-K，默认 1")
    parser.add_argument("--min-guard-recall", type=float, default=0.9, help="注入拦截召回率下限")
    parser.add_argument("--max-guard-fpr", type=float, default=0.1, help="注入防护误伤率上限")
    parser.add_argument("--min-intent-acc", type=float, default=0.8, help="规则路由准确率下限")
    parser.add_argument("--min-retrieval-hit", type=float, default=0.8, help="检索命中率下限")
    parser.add_argument("--report", default=os.path.join(EVAL_DIR, "report.md"), help="报告输出路径")
    args = parser.parse_args()

    guard = eval_guard()
    intent = eval_intent(args.with_llm)
    retrieval = eval_retrieval(args.top_k)

    _print_guard(guard)
    _print_intent(intent)
    _print_retrieval(retrieval)
    _write_report(args.report, guard, intent, retrieval)
    print(f"\n报告已写入: {args.report}")

    failures: List[str] = []
    if guard["recall"] < args.min_guard_recall:
        failures.append(f"注入召回率 {guard['recall']:.1%} < 阈值 {args.min_guard_recall:.1%}")
    if guard["false_positive_rate"] > args.max_guard_fpr:
        failures.append(f"注入误伤率 {guard['false_positive_rate']:.1%} > 阈值 {args.max_guard_fpr:.1%}")
    if intent["rule_only_accuracy"] < args.min_intent_acc:
        failures.append(
            f"规则路由准确率 {intent['rule_only_accuracy']:.1%} < 阈值 {args.min_intent_acc:.1%}"
        )
    if retrieval["hit_rate"] < args.min_retrieval_hit:
        failures.append(
            f"检索命中率 {retrieval['hit_rate']:.1%} < 阈值 {args.min_retrieval_hit:.1%}"
        )

    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1
    print("[PASS] 所有评估项均达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
