cd /Users/aniruddha/Documents/research/qbranch
uv run python src/run_build.py --dataset musique --n 100 --variant qbranch > runs/logs/build_musique100_qbranch.log 2>&1 &
uv run python src/run_build.py --dataset musique --n 100 --variant openie > runs/logs/build_musique100_openie.log 2>&1 &
wait
uv run python src/run_answer.py --dataset musique --n 100 > runs/logs/answer_musique100.log 2>&1
echo PILOT_DONE >> runs/logs/answer_musique100.log
