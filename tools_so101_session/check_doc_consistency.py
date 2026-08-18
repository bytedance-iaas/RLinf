# STATUS: TOOL — 通用工具，与具体阶段无关。 核对文档里的每个名字是否与仓库/磁盘一致
"""Cross-check every name a pipeline document mentions against the repository.

Why this exists: the documents name four different KINDS of thing that all look
alike -- registry entries (pi05_so101_v4), training configs (so101_sft_v8),
result directories (so101_ppo_v13) and datasets (so101-sim-demos-v9). A reader
cannot tell them apart by shape, and neither can the author from memory, so the
tables drift from the code and from each other. Reviewing them by eye has now
missed the same class of error several times.

So this checks them mechanically, by category:

  registry entry   -> name="..." in rlinf/models/embodiment/openpi/dataconfig/__init__.py
  training config  -> examples/{sft,embodiment}/config/<name>.yaml
  dataset          -> $HF_LEROBOT_HOME/<name>/meta/info.json
  result checkpoint-> the path exists on disk
  script           -> tools_so101_session/<name>
  assets stats     -> the norm_stats.json exists

and separately checks the document's own summary table both ways: every entry it
tells you to use must be listed there, and every entry listed must actually be
used. That is the defect that made §1.4 claim six entries while stage G used a
seventh.

    python tools_so101_session/check_doc_consistency.py SO101_PIPELINE_ZH.md

Exit code is non-zero if anything fails, so it can gate a commit.
"""

import os
import re
import sys

REPO = "/data08/henryg/pai/RLinf"
DATA = os.environ.get("HF_LEROBOT_HOME", "/data08/henryg/pai/data")
REGISTRY = os.path.join(REPO, "rlinf/models/embodiment/openpi/dataconfig/__init__.py")


def known_registry_entries():
    src = open(REGISTRY, encoding="utf-8").read()
    return set(re.findall(r'name="([a-z0-9_]+)"', src))


def config_exists(name):
    return any(
        os.path.isfile(os.path.join(REPO, d, f"{name}.yaml"))
        for d in ("examples/sft/config", "examples/embodiment/config")
    )


def split_blocks(text):
    """Return (prose, commands). Code fences are commands."""
    prose, cmd, inside = [], [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            inside = not inside
            continue
        (cmd if inside else prose).append(line)
    return "\n".join(prose), "\n".join(cmd)


def main():
    doc = sys.argv[1] if len(sys.argv) > 1 else "SO101_PIPELINE_ZH.md"
    text = open(os.path.join(REPO, doc), encoding="utf-8").read()
    prose, commands = split_blocks(text)
    entries = known_registry_entries()
    fails = []

    def check(kind, name, ok, hint=""):
        if not ok:
            fails.append(f"{kind:16} {name:34} {hint}")

    # 1. registry entries
    for n in sorted(set(re.findall(r'\bpi05_so101(?:_[a-z0-9]+)?\b', text))):
        check("注册表条目", n, n in entries, "不在 dataconfig/__init__.py 里")

    # 2. training configs -- only names passed to --config-name / config_name=
    used_cfgs = set(re.findall(r'--config-name[= ]+([a-z0-9_]+)', text))
    used_cfgs |= set(re.findall(r'CFG=([a-z0-9_]+)', text))
    for n in sorted(used_cfgs):
        if n.startswith("pi05_"):
            check("注册表条目", n, n in entries, "作为 --config-name 传给了 openpi")
        else:
            check("训练/评测配置", n, config_exists(n), "examples/*/config/ 下没有这个 yaml")

    # 3. datasets
    for n in sorted(set(re.findall(r'\bso101-(?:sim-demos|cotrain|pick-place)[a-z0-9-]*\b', text))):
        if n == "so101-pick-place-v2":       # HF repo id, materialised under another name here
            continue
        check("数据集", n, os.path.isfile(os.path.join(DATA, n, "meta", "info.json")),
              f"{DATA}/{n} 下没有 meta/info.json")

    # 4. absolute paths that must exist. log_path targets are OUTPUTS -- the run
    #    creates them -- so they are excluded rather than reported as missing.
    outputs = set(re.findall(r'log_path=(/data08/[A-Za-z0-9_./-]+)', text))
    for p in sorted(set(re.findall(r'/data08/henryg/pai/(?:results|data)/[A-Za-z0-9_./-]+', text))):
        if "*" in p or p.endswith("/") or "<" in p or p in outputs:
            continue
        check("路径", p[-34:], os.path.exists(p), "磁盘上不存在")

    # 5. scripts
    for n in sorted(set(re.findall(r'tools_so101_session/([A-Za-z0-9_]+\.(?:py|sh))', text))):
        check("脚本", n, os.path.isfile(os.path.join(REPO, "tools_so101_session", n)), "文件不存在")

    # 6. norm_stats files
    for p in sorted(set(re.findall(r'assets/[A-Za-z0-9_/-]+/norm_stats\.json', text))):
        check("统计量", p[-34:], os.path.isfile(os.path.join(REPO, p)), "文件不存在")

    # 7. the drift that started this. The rule is not "every mention" -- a caveat
    #    may legitimately name an abandoned entry -- but "every entry the document
    #    tells you to USE as a config_name must appear in its own summary table".
    #    §1.4 used to be a second, overlapping table that claimed to be complete at
    #    six while stage G's parameter table used a seventh (pi05_so101_v14).
    # Collect the entries the document tells you to pass. Three shapes, each
    # matched explicitly rather than by proximity -- a "within N characters of
    # config_name" rule silently depends on how a table row happens to be worded.
    used_as_cfg = set(re.findall(r'--config-name[=\s]+(pi05_so101(?:_[a-z0-9]+)?)\b', text))
    used_as_cfg |= set(re.findall(r'config_name\s*=\s*(pi05_so101(?:_[a-z0-9]+)?)\b', text))
    for row in text.split("\n"):                       # table rows naming an entry
        if row.startswith("|") and ("config_name" in row or "条目" in row):
            used_as_cfg |= set(re.findall(r'`(pi05_so101(?:_[a-z0-9]+)?)`', row))
    # Only the pipeline document carries this table; the others legitimately have
    # none, so the two-way check is skipped rather than failed for them.
    # Match on the heading's TEXT, not its number -- another document's §1.5 is a
    # different section entirely, and matching by number silently compared against it.
    table = re.search(r'### [0-9.]+ 各阶段产物一览.*?(?=\n## )', text, re.S)
    if table:
        listed = set(re.findall(r'\bpi05_so101(?:_[a-z0-9]+)?\b', table.group(0)))
        for n in sorted(used_as_cfg - listed):
            fails.append(f"{'表格遗漏':16} {n:34} 文档让你用它，但 §1.5 的表没列")
        for n in sorted(listed - used_as_cfg):
            fails.append(f"{'表格多余':16} {n:34} §1.5 列了，但文档里没有一处让你用它")
    else:
        print("  （本文档没有 §1.5 产物表，跳过双向核对）")

    # 8. Every script the document TELLS YOU TO RUN must also appear in its tool
    #    list. so101_smoke.py was added to the install section and left out of §2,
    #    so a reader meeting it mid-document had nowhere to look it up.
    tools = re.search(r'## 2\. 工具清单.*?(?=\n## 3\.)', text, re.S)
    if tools:
        invoked = set(re.findall(r'tools_so101_session/([A-Za-z0-9_]+\.(?:py|sh))', commands))
        indexed = set(re.findall(r'`([A-Za-z0-9_]+\.(?:py|sh))`', tools.group(0)))
        for n in sorted(invoked - indexed):
            fails.append(f"{'清单遗漏':16} {n:34} 命令里让你跑它，但 §2 工具清单没列")

    print(f"检查 {doc}")
    print(f"  注册表条目：仓库里 {len(entries)} 个，本文档用 {len(used_as_cfg)} 个")
    src = open(REGISTRY, encoding="utf-8").read()
    for n in sorted(used_as_cfg):
        m = re.search(r'name="%s",.*?repo_id="([^"]+)"' % n, src, re.S)
        print(f"      {n:18} -> repo_id={m.group(1) if m else '?'}")
    if fails:
        print(f"\n❌ {len(fails)} 处不一致：")
        for f in fails:
            print("   " + f)
        sys.exit(1)
    print("\n✅ 文档里的每个名字都与仓库/磁盘一致")


if __name__ == "__main__":
    main()
