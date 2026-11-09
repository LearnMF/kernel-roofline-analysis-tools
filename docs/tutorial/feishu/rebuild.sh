#!/bin/bash
# rebuild.sh <doc_id> <part1.xml> [img:<path>|<caption>] [part2.xml] ...
# 第一个 xml 用 overwrite，之后的 xml 用 append，img: 项用 media-insert（追加到末尾）。
# 在 docs/tutorial 目录下运行（图片路径相对该目录）。
set -euo pipefail
export LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1
doc=$1; shift
first=1
for item in "$@"; do
  if [[ $item == img:* ]]; then
    spec=${item#img:}; path=${spec%%|*}; cap=${spec#*|}
    lark-cli docs +media-insert --as user --doc "$doc" --file "$path" --width 860 --caption "$cap" 2>&1 | grep -q '"ok": true' && echo "img ok $path" || { echo "img FAIL $path"; exit 1; }
  else
    cmd=append; [[ $first == 1 ]] && cmd=overwrite; first=0
    out=$(lark-cli docs +update --as user --doc "$doc" --command $cmd --content "@feishu/$item" 2>&1)
    echo "$out" | grep -q '"result": "success"' && echo "$cmd ok $item" || { echo "$cmd FAIL $item"; echo "$out" | tail -20; exit 1; }
  fi
done
