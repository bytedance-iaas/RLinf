# STATUS: TOOL — 通用工具，与具体阶段无关。 把 SKILL.md 的"好"定义成判据并一次列全
"""Lint a skill document against what makes it general and usable.

Why this exists: the skill was improved by a point-fix loop -- a reader names a
defect, the author fixes that one instance, the reader finds the next of the same
kind. Six rounds of that. The skill's own section 9b says to enumerate the class
before fixing anything, which requires the class to be expressible as a check.
This is that check.

Nine predicates, in two groups.

MECHANICAL (this file decides):
  1  no project/robot proper nouns in the rules
  2  no internal run identifiers in the rules
  3  no non-English quotation in the rules
  4  no dates in the rules (they invite discounting a rule by age)
  5  section numbers unique and in ascending order
  6  every section cross-reference resolves
  7  every rule section is reachable from the entry section
  8  appendices contain all project-specific content, and rules contain none

JUDGEMENT (this file only PRESENTS; a human or model decides):
  9  every claim carries a mechanism and evidence, and the evidence is legible
     without project context -- printed as a list so the review is bounded,
     instead of re-reading the document linearly, which is what kept failing.

    python tools_so101_session/check_skill.py [path] [--claims]

Exit code is non-zero if any mechanical predicate fails.
"""

import argparse
import os
import re
import sys

DEFAULT = ".claude/skills/rlinf-embodied-training/SKILL.md"

# Proper nouns that belong to one project. Extend per project; the point is that
# the list is explicit rather than remembered.
PROJECT_NOUNS = r"SO101|so101|pi05_so101|maniskill_so101|GrabRedCube|henry-guo"
RUN_IDS = r"\b(?:pp\d[a-z]?|v\d+[a-z]?)\b"
CJK = r"[一-鿿]"
DATES = r"\b20\d\d-\d\d(?:-\d\d)?\b"


def split_rules_and_appendix(text):
    i = text.find("## Appendix")
    return (text, "") if i < 0 else (text[:i], text[i:])


def sections(text):
    """Return [(id, title, body)] for '## <id>. <title>' headings."""
    out = []
    for block in re.split(r"\n(?=## )", text):
        m = re.match(r"## ([0-9]+[a-z]?)\. (.*)", block)
        if m:
            out.append((m.group(1), m.group(2).strip(), block))
    return out


def subsection_ids(text):
    """'### 9a. ...' subsections are referenceable too, and a linter that only
    knows about '##' reports every reference to one as dangling."""
    return {m.group(1) for m in re.finditer(r"^### ([0-9]+[a-z]?)[.\s]", text, re.M)}


def sort_key(sid):
    m = re.match(r"([0-9]+)([a-z]?)", sid)
    return (int(m.group(1)), m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--claims", action="store_true",
                    help="print the claim/evidence list for the judgement pass")
    args = ap.parse_args()

    text = open(args.path, encoding="utf-8").read()
    rules, appendix = split_rules_and_appendix(text)
    secs = sections(rules)
    fails = []

    def scan(pattern, label, hint):
        for i, line in enumerate(rules.split("\n"), 1):
            # A line may opt out with an inline marker, which keeps the exception
            # visible in the document instead of hidden in this file's logic.
            if "lint-ok" in line:
                continue
            for m in re.finditer(pattern, line):
                fails.append((i, label, m.group(0), hint, line.strip()[:90]))

    # 1-4: things that make a rule non-transferable
    scan(PROJECT_NOUNS, "项目专名", "换成通用名词（the policy / the object / the training region）")
    scan(RUN_IDS, "运行编号", "改成这次运行扮演的角色（'a run with 1 update/epoch'）")
    scan(CJK, "非英文引语", "改写成实际观察到了什么，而不是引用某人的话")
    scan(DATES, "日期", "规则不该带日期；快照才需要，且用 git 更可靠")

    # 5: numbering unique and ordered
    ids = [s[0] for s in secs]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        fails.append((0, "章节重号", ",".join(dupes), "同号章节读者无法引用", ""))
    if ids != sorted(ids, key=sort_key):
        fails.append((0, "章节乱序", "", "追加式增长会让读者顺着读时跳过整段", ""))

    # 6: cross-references resolve
    referenceable = set(ids) | subsection_ids(text)
    for ref in sorted(set(re.findall(r"§([0-9]+[a-z]?)(?![0-9a-z])", text))):
        if ref not in referenceable:
            fails.append((0, "悬空引用", f"§{ref}", "没有这个章节", ""))

    # 7: every rule section reachable from the entry section
    entry = next((b for i, _, b in secs if i == "0"), None)
    if entry is None:
        fails.append((0, "缺少入口", "§0", "没有'先做什么'的入口，读者无法开始", ""))
    else:
        named = set(re.findall(r"§([0-9]+[a-z]?)(?![0-9a-z])", entry))
        # a section counts as reachable if the entry names it or names its group
        orphans = [i for i in ids
                   if i != "0" and i not in named and re.match(r"[0-9]+", i).group(0) not in named]
        if orphans:
            fails.append((0, "入口未覆盖", ",".join(orphans),
                          "§0 没有把这些章节挂进任何一道门", ""))

    # 8: the appendix should hold the project specifics
    if appendix and not re.search(PROJECT_NOUNS, appendix):
        fails.append((0, "附录可疑", "", "附录里没有任何项目专名 —— 它还是项目快照吗？", ""))

    print(f"检查 {args.path}")
    print(f"  规则 {len(rules.splitlines())} 行 / 附录 {len(appendix.splitlines())} 行，"
          f"{len(secs)} 个章节")
    if fails:
        print(f"\n❌ {len(fails)} 处：")
        for ln, label, hit, hint, ctx in fails:
            where = f"L{ln}" if ln else "  —"
            print(f"   {where:>6}  {label:8} {hit[:22]:24} {hint}")
            if ctx:
                print(f"           {ctx}")
    else:
        print("\n✅ 机械判据全部通过")

    if args.claims:
        print("\n" + "=" * 78)
        print("判断项：逐条读下面的 claim + evidence，问两个问题——")
        print("  (a) 换一个机器人/任务，这条还成立吗？")
        print("  (b) 不了解这个项目的人，读得懂这条证据吗？")
        print("=" * 78)
        for sid, title, body in secs:
            claims = re.findall(r"^\*\*(.+?)\*\*", body, re.M)
            evid = re.findall(r"^> Evidence:(.*)$", body, re.M)
            print(f"\n§{sid} {title[:64]}   [{len(claims)} claim / {len(evid)} evidence]")
            for c in claims:
                print(f"    · {c[:96]}")
            for e in evid:
                print(f"      ↳ {e.strip()[:96]}")
            # Only cry wolf when a section makes claims and offers nothing to
            # check them against -- neither an Evidence line, nor an inline
            # measurement, nor a citation. An index section legitimately has none.
            grounded = bool(evid) or re.search(r"\d+(\.\d+)?\s?%|arXiv|`[a-z_]+\.(py|yaml)`", body)
            if claims and not grounded and sid != "0":
                print("      ⚠ 有主张，但既无 Evidence 行也无可核对的数字/引用")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    os.chdir("/data08/henryg/pai/RLinf")
    main()
