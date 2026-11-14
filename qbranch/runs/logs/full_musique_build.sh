#!/bin/zsh
# Full MuSiQue (HippoRAG 2 setting: 1,000 questions, 11,656 passages), 10 stream batches.
cd /Users/aniruddha/Documents/research/qbranch
uv run python src/run_build.py --dataset musique --n 0 --variant qbranch --batches 10 --workers 16 > runs/logs/build_musique_all_qbranch.log 2>&1
uv run python src/run_build.py --dataset musique --n 0 --variant openie --batches 10 --workers 16 > runs/logs/build_musique_all_openie.log 2>&1
echo BUILD_DONE
