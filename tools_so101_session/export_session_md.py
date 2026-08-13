"""Export the full session transcript (jsonl) to one Markdown file, verbatim:
conversation turns, tool calls with inputs, tool results, plus an appendix with
the CURRENT full contents of every project file this session created/modified
and every generated script.
"""
import json
import os

SRC = "/root/.claude/projects/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd.jsonl"
OUT = "/data08/henryg/pai/RLinf/SO101_SESSION_LOG.md"

APPENDIX_FILES = [
    # in-tree code created/modified this session
    "/data08/henryg/pai/RLinf/rlinf/envs/maniskill/tasks/so101_pick_place.py",
    "/data08/henryg/pai/RLinf/rlinf/envs/maniskill/so101_agent.py",
    "/data08/henryg/pai/RLinf/rlinf/envs/maniskill/so101_calib.py",
    "/data08/henryg/pai/RLinf/rlinf/envs/maniskill/maniskill_env.py",
    "/data08/henryg/pai/RLinf/rlinf/scheduler/collective/collective_group.py",
    "/data08/henryg/pai/RLinf/rlinf/scheduler/cluster/cluster.py",
    "/data08/henryg/pai/RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py",
    "/data08/henryg/pai/RLinf/rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py",
    "/data08/henryg/pai/RLinf/rlinf/models/embodiment/openpi/policies/so101_policy.py",
    "/data08/henryg/pai/RLinf/toolkits/preflight_config.py",
    "/data08/henryg/pai/RLinf/toolkits/invariant_audit.py",
    "/data08/henryg/pai/RLinf/toolkits/so101_to_usd.py",
    # configs
    "/data08/henryg/pai/RLinf/examples/embodiment/config/env/maniskill_so101_pick_place.yaml",
    "/data08/henryg/pai/RLinf/examples/embodiment/config/so101_eval_openpi_pi05.yaml",
    "/data08/henryg/pai/RLinf/examples/embodiment/config/so101_ppo_openpi_pi05.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v3.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v4.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v5.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v7.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v8.yaml",
    "/data08/henryg/pai/RLinf/examples/sft/config/so101_sft_v9.yaml",
    "/data08/henryg/pai/RLinf/V8_COMMANDS.md",
    "/data08/henryg/pai/RLinf/V8_COMMANDS_ZH.md",
    "/data08/henryg/pai/RLinf/examples/embodiment/config/so101_ppo_v6_official.yaml",
    "/data08/henryg/pai/RLinf/SO101_PP_80PCT_RUNBOOK.md",
    # skill
    "/data08/henryg/pai/RLinf/.claude/skills/rlinf-embodied-training/SKILL.md",
]
SCRATCH = "/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad"


def block_to_md(block):
    t = block.get("type")
    if t == "text":
        return block.get("text", "")
    if t == "thinking":
        return "> *(internal reasoning omitted from rendering; present in raw jsonl)*"
    if t == "tool_use":
        name = block.get("name", "?")
        inp = json.dumps(block.get("input", {}), ensure_ascii=False, indent=2)
        return f"**[TOOL CALL: {name}]**\n```json\n{inp}\n```"
    if t == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif isinstance(c, dict) and c.get("type") == "image":
                    parts.append("*(image result)*")
            body = "\n".join(parts)
        else:
            body = str(content)
        return f"**[TOOL RESULT]**\n```\n{body}\n```"
    return f"*({t})*"


with open(OUT, "w") as out:
    out.write("# SO101 + PI0.5 Training — Full Session Log\n\n")
    out.write("Session id: 3e748c24-1f70-49ee-a01c-395d2f1161dd — exported 2026-08-11.\n\n")
    out.write("Part 1 renders every recorded turn of the session transcript verbatim "
              "(user turns, assistant turns, tool calls and their results). "
              "Part 2 appends the CURRENT full contents of all files created or "
              "modified during the session, and Part 3 the generated scripts.\n\n---\n\n")
    out.write("## Part 1 — Conversation transcript\n\n")
    n = 0
    with open(SRC) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            etype = e.get("type")
            if etype not in ("user", "assistant"):
                continue
            msg = e.get("message", {})
            role = msg.get("role", etype)
            content = msg.get("content")
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            if isinstance(content, str):
                body = content
            elif isinstance(content, list):
                body = "\n\n".join(block_to_md(b) for b in content if isinstance(b, dict))
            else:
                continue
            if not body.strip():
                continue
            label = "USER" if role == "user" else "ASSISTANT"
            out.write(f"### [{ts}] {label}\n\n{body}\n\n---\n\n")
            n += 1
    out.write(f"\n*({n} turns rendered)*\n\n")
    out.write("## Part 2 — Current contents of files created/modified this session\n\n")
    for p in APPENDIX_FILES:
        if not os.path.exists(p):
            out.write(f"### {p}\n\n*(missing)*\n\n")
            continue
        lang = "python" if p.endswith(".py") else ("yaml" if p.endswith(".yaml") else "markdown")
        out.write(f"### {p}\n\n```{lang}\n{open(p, errors='replace').read()}\n```\n\n")
    out.write("## Part 3 — Generated scripts (scratchpad)\n\n")
    for fn in sorted(os.listdir(SCRATCH)):
        if fn.endswith((".py", ".sh")):
            p = os.path.join(SCRATCH, fn)
            lang = "python" if fn.endswith(".py") else "bash"
            out.write(f"### scratchpad/{fn}\n\n```{lang}\n{open(p, errors='replace').read()}\n```\n\n")

print("written:", OUT, os.path.getsize(OUT) // 1024, "KB")
