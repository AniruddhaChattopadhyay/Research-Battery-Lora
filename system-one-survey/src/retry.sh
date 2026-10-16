#!/usr/bin/env bash
# Run one command several times, for a bad network connection.
# The runners skip the calls that already returned status 200, so each pass sends only
# the missing calls. Before each pass, wait until OpenRouter answers.
#
#   bash src/retry.sh <passes> <command> [arguments ...]
passes=$1
shift
for i in $(seq 1 "$passes"); do
  until curl -s -o /dev/null -m 15 https://openrouter.ai/api/v1/models?output_modalities=decisions; do
    echo "$(date -u +%FT%TZ) no network, wait 60 s"
    sleep 60
  done
  echo "$(date -u +%FT%TZ) pass $i of $passes"
  "$@"
  sleep 20
done
echo "$(date -u +%FT%TZ) all passes done"
