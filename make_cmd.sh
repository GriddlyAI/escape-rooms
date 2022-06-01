#!/bin/bash
# debug=
debug=echo
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

tag=${1:-Grafter-June-1}
declare -a lrs=("1e-4" "2.5e-4" "5e-4")
declare -a nsteps=("128" "256")
declare -a num_minibatches=("4" "8")
declare -a seeds=("1" "2" "3" "4" "5" "6")

# run parallel
count=0
for lr in "${lrs[@]}"; do
    for nstep in "${nsteps[@]}"; do
        for n_mini in "${num_minibatches[@]}"; do
            for seed in "${seeds[@]}"; do
                group="${tag}-${lr}-${nstep}-${n_mini}"
                echo python3 ppo.py --learning-rate "$lr" --num-steps "$nstep" --num-minibatches "$n_mini" --wandb-group "$group" --seed "$seed"
            done
        done
    done
done
