#!/usr/bin/env bash
# demo-sequence.sh — the actual commands shown in the demo recording
# Called by record-demo.sh inside asciinema rec --command

# Activate venv silently
source /Users/nameetpotnis/Projects/pdfmux/.venv/bin/activate 2>/dev/null

# Typing simulator
type_cmd() {
  local cmd="$1"
  for (( i=0; i<${#cmd}; i++ )); do
    printf '%s' "${cmd:$i:1}"
    sleep 0.04
  done
}

prompt() {
  printf '\033[1;32m$\033[0m '
}

# --- Scene 1: Check setup ---
sleep 0.8
prompt
type_cmd "pdfmux doctor"
echo
sleep 0.3
pdfmux doctor
sleep 2

# --- Scene 2: Convert a PDF ---
prompt
type_cmd "pdfmux convert demo-sample.pdf"
echo
sleep 0.3
cd /Users/nameetpotnis/Projects/pdfmux
pdfmux convert demo-sample.pdf
sleep 1.5

# --- Scene 3: Analyze per-page quality ---
prompt
type_cmd "pdfmux analyze demo-sample.pdf"
echo
sleep 0.3
pdfmux analyze demo-sample.pdf
sleep 2

# --- Scene 4: Show the output ---
prompt
type_cmd "head -8 demo-sample.md"
echo
sleep 0.3
head -8 demo-sample.md
sleep 2

echo
