#!/usr/bin/env bash
# record-demo.sh — scripted terminal demo for pdfmux README
# Uses asciinema + expect-style typing for a clean, repeatable recording.
#
# Usage:
#   cd /Users/nameetpotnis/Projects/pdfmux
#   source .venv/bin/activate
#   bash scripts/record-demo.sh
#
# Output: demo.cast (asciicast v2)
# Then convert:  svg-term --in demo.cast --out demo.svg --window --no-cursor --width 80 --height 24

set -e

CAST_FILE="/Users/nameetpotnis/Projects/pdfmux/demo.cast"
PDF_FILE="/Users/nameetpotnis/Projects/pdfmux/demo-sample.pdf"

# Typing simulator: prints chars one at a time with small delays
type_cmd() {
  local cmd="$1"
  for (( i=0; i<${#cmd}; i++ )); do
    printf '%s' "${cmd:$i:1}"
    sleep 0.04
  done
  echo
}

# Run a command with simulated typing, then execute
run_typed() {
  local cmd="$1"
  local pause_after="${2:-1.5}"
  printf '$ '
  type_cmd "$cmd"
  sleep 0.3
  eval "$cmd"
  sleep "$pause_after"
}

echo "Recording to $CAST_FILE ..."

# Use asciinema rec with a command that runs our demo sequence
/Users/nameetpotnis/Library/Python/3.9/bin/asciinema rec \
  --overwrite \
  --cols 80 \
  --rows 24 \
  --title "pdfmux — PDF extraction that checks its own work" \
  --command "bash /Users/nameetpotnis/Projects/pdfmux/scripts/demo-sequence.sh" \
  "$CAST_FILE"

echo ""
echo "Recorded: $CAST_FILE"
echo ""
echo "Convert to SVG:"
echo "  svg-term --in $CAST_FILE --out demo.svg --window --no-cursor --width 80 --height 24"
echo ""
echo "Or upload to asciinema.org:"
echo "  asciinema upload $CAST_FILE"
