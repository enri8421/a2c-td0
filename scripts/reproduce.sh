#!/bin/sh
# Reproduce seven reference runs; use another prefix to run an independent repeat.
set -eu
cd "$(dirname "$0")/.."

prefix=${1:-reference}
output=${2:-results}
case "$prefix" in
    ''|*[!A-Za-z0-9_-]*) echo "Use letters, numbers, underscores, or hyphens for the prefix." >&2; exit 1 ;;
esac
cartpole="$prefix-cartpole-cautious-s23-1m"
acrobot="$prefix-acrobot-balanced-s11-750k"
mountaincar="$prefix-mountaincar-tanh-s0-3m"
lunarlander="$prefix-lunarlander-explore-s61-6m"
frozenlake="$prefix-frozenlake-balanced-s11-1m"
taxi="$prefix-taxi-explore-s47-1m"
cliffwalking="$prefix-cliffwalking-starts-s2-100k"
for name in "$cartpole" "$acrobot" "$mountaincar" "$lunarlander" "$frozenlake" "$taxi" "$cliffwalking"; do
    if [ -e "runs/$name" ]; then
        echo "Run already exists: $name. Choose a different prefix; nothing was overwritten." >&2
        exit 1
    fi
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
uv sync --locked

train_and_analyse() {
    name=$1
    shift
    uv run --locked python -m experiments.train --run-name "$name" --device cpu "$@"
    uv run --locked python -m experiments.analyse --run-name "$name" --device cpu --episodes 1000 --record
}

train_and_analyse "$mountaincar" --env MountainCar-v0 --seed 0 --total-steps 3000000 \
    --lr-value 0.0005 --lr-policy 0.0002 --gamma 0.99 --beta 0.001 \
    --adam-eps 0.00001 --scale-observations --max-episode-steps 1000

train_and_analyse "$cliffwalking" --env CliffWalking-v1 --seed 2 --total-steps 100000 \
    --lr-value 0.001 --lr-policy 0.001 --gamma 0.99 --beta 0.01 \
    --no-scale-observations --max-episode-steps 200 \
    --adam-eps 0.001

train_and_analyse "$cartpole" --env CartPole-v1 --seed 23 --total-steps 1000000 \
    --lr-value 0.0005 --lr-policy 0.0002 --gamma 0.99 --beta 0.001 \
    --no-scale-observations --max-episode-steps 500 \
    --adam-eps 0.001

train_and_analyse "$acrobot" --env Acrobot-v1 --seed 11 --total-steps 750000 \
    --lr-value 0.001 --lr-policy 0.0005 --gamma 0.99 --beta 0.01 \
    --no-scale-observations --max-episode-steps 500 \
    --adam-eps 0.001

train_and_analyse "$lunarlander" --env LunarLander-v3 --seed 61 --total-steps 6000000 \
    --lr-value 0.001 --lr-policy 0.0005 --gamma 0.99 --beta 0.05 \
    --no-scale-observations --max-episode-steps 1000 \
    --adam-eps 0.001

train_and_analyse "$frozenlake" --env FrozenLake-v1 --seed 11 --total-steps 1000000 \
    --lr-value 0.001 --lr-policy 0.001 --gamma 0.99 --beta 0.01 \
    --no-scale-observations --max-episode-steps 100 \
    --adam-eps 0.001

train_and_analyse "$taxi" --env Taxi-v4 --seed 47 --total-steps 1000000 \
    --lr-value 0.003 --lr-policy 0.003 --gamma 0.99 --beta 5 \
    --no-scale-observations --max-episode-steps 200 \
    --adam-eps 0.001

uv run --locked python -m experiments.scores \
    "$cartpole" "$acrobot" "$mountaincar" "$lunarlander" "$frozenlake" "$taxi" "$cliffwalking" --output "$output"
