from __future__ import annotations

import os
from openai import OpenAI

# 建议：用环境变量读 key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
你是“中文字幕受约束润色器”。

任务：将输入的中文字幕改写为更自然、更口语、更适合字幕展示的中文。

硬性约束：
1) 禁止新增信息。
2) 禁止删除关键信息。
3) 禁止改动任何数字、日期、金额、单位、专有名词、术语。
4) 保持原意、语气与事实关系不变（否定/比较/因果不得改变）。
5) 输出尽量简短，适合字幕展示。

仅输出润色后的中文，不要解释。
""".strip()


def openai_rewrite_text(text: str) -> str:
    """
    受约束字幕润色（单句/单段）。
    """
    # 你也可以换成 gpt-4o-mini 以降低成本
    # GPT-5 mini 强指令遵循、速度快，适合这类任务。:contentReference[oaicite:2]{index=2}
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ],
    )

    # SDK 提供的便捷字段（兼容输出结构）
    out = getattr(response, "output_text", None)
    if out:
        return out.strip()

    # 兜底：按通用结构取
    try:
        return response.output[0].content[0].text.strip()
    except Exception:
        return text
