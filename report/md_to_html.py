#!/usr/bin/env python3
"""把实验报告 Markdown 转成可浏览器打开/打印的 HTML。"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "experiment_report.md"
OUT = ROOT / "experiment_report.html"


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    in_table = False
    in_quote = False

    def flush_quote() -> None:
        nonlocal in_quote
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_quote()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                out.append("<pre><code>")
                in_code = True
            i += 1
            continue
        if in_code:
            out.append(html.escape(line))
            i += 1
            continue

        img = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if img:
            flush_quote()
            alt, src = img.group(1), img.group(2)
            out.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>"
            )
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1]):
            flush_quote()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if re.match(r"^\|?\s*:?-+", lines[i]):
                    i += 1
                    continue
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            out.append("<table>")
            for ridx, cells in enumerate(rows):
                tag = "th" if ridx == 0 else "td"
                out.append(
                    "<tr>"
                    + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells)
                    + "</tr>"
                )
            out.append("</table>")
            continue

        if line.startswith("> "):
            if not in_quote:
                out.append("<blockquote>")
                in_quote = True
            out.append(f"<p>{inline(line[2:])}</p>")
            i += 1
            continue
        flush_quote()

        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("#### "):
            out.append(f"<h4>{inline(line[5:])}</h4>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                item = re.sub(r"^\d+\. ", "", lines[i])
                items.append(f"<li>{inline(item)}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip() == "":
            pass
        else:
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    flush_quote()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def main() -> None:
    body = convert(MD.read_text(encoding="utf-8"))
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>实验三：基于卷积神经网络的服装图像分类</title>
<style>
body {{ font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", serif;
       max-width: 980px; margin: 32px auto; padding: 0 20px;
       line-height: 1.65; color: #222; }}
h1,h2,h3 {{ font-family: "Noto Sans CJK SC", sans-serif; }}
h1 {{ font-size: 1.7rem; }}
h2 {{ border-bottom: 2px solid #3182bd; padding-bottom: 4px; margin-top: 2em; }}
figure {{ margin: 1.2em 0; text-align: center; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
figcaption {{ font-size: 0.9rem; color: #555; margin-top: 0.4em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
th {{ background: #f0f4f8; }}
pre {{ background: #111; color: #eee; padding: 12px; overflow-x: auto;
      font-size: 0.85rem; }}
blockquote {{ background: #eef6fb; border-left: 4px solid #3182bd;
              padding: 8px 16px; }}
code {{ font-family: ui-monospace, monospace; }}
p code, li code, td code {{ background: #f4f4f4; padding: 0 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
