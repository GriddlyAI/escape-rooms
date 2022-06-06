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

tag=${1:-Grafter-initial}
date=`date +%s`
declare -a lrs=("1e-4" "2.5e-4" "5e-4")
declare -a lrs_aneal=("True" "False")
declare -a nsteps=("128" "256")
declare -a num_minibatches=("4" "8")
declare -a seeds=("1" "2" "3" "4" "5" "6")

# run parallel
count=0
for lr in "${lrs[@]}"; do
    for nstep in "${nsteps[@]}"; do
        for n_mini in "${num_minibatches[@]}"; do
          for lr_an in "${lrs_aneal[@]}"; do
            for seed in "${seeds[@]}"; do
                group="${tag}-${lr}-${nstep}-${n_mini}-${lr_an}"
                echo python3 ppo.py --learning-rate "$lr" --anneal-lr "$lr_an" --num-steps "$nstep" --num-minibatches "$n_mini" --wandb-group "$group" --seed "$seed" --exp-name "$tag"
            done
          done
        done
    done
done
