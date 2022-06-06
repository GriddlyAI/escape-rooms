#!/bin/sh
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -N escape-rooms-sweep-SPS
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -l h_rt=1:0:0
#$ -t 1-25
#$ -o logs/
#$ -e logs/

exp_name_values=( escape-rooms-sweep-SPS )
track_values=( True )
cuda_values=( True )
total_timesteps_values=( 1000000 )
num_envs_values=( 32 64 128 256 512 )
num_steps_values=( 32 64 128 256 512 )
num_minibatches_values=( 32 )
learning_rate_values=( 0.05 )
ent_coef_values=( 0.1 )
data_dir_values=( /data/scratch/acw434/escape-rooms-sweep-SPS )
trial=${SGE_TASK_ID}
exp_name="${exp_name_values[$(( trial % ${#exp_name_values[@]} ))]}"
trial=$(( trial / ${#exp_name_values[@]} ))
track="${track_values[$(( trial % ${#track_values[@]} ))]}"
trial=$(( trial / ${#track_values[@]} ))
cuda="${cuda_values[$(( trial % ${#cuda_values[@]} ))]}"
trial=$(( trial / ${#cuda_values[@]} ))
total_timesteps="${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}"
trial=$(( trial / ${#total_timesteps_values[@]} ))
num_envs="${num_envs_values[$(( trial % ${#num_envs_values[@]} ))]}"
trial=$(( trial / ${#num_envs_values[@]} ))
num_steps="${num_steps_values[$(( trial % ${#num_steps_values[@]} ))]}"
trial=$(( trial / ${#num_steps_values[@]} ))
num_minibatches="${num_minibatches_values[$(( trial % ${#num_minibatches_values[@]} ))]}"
trial=$(( trial / ${#num_minibatches_values[@]} ))
learning_rate="${learning_rate_values[$(( trial % ${#learning_rate_values[@]} ))]}"
trial=$(( trial / ${#learning_rate_values[@]} ))
ent_coef="${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}"
trial=$(( trial / ${#ent_coef_values[@]} ))
data_dir="${data_dir_values[$(( trial % ${#data_dir_values[@]} ))]}"

module purge
module load cuda anaconda3 vulkan-sdk
conda activate conda_poetry_base

export PYTHONUNBUFFERED=1

# Set up poetry
cd ~/escape-rooms
poetry shell

python ~/escape-rooms/ppo.py  --exp-name="${exp_name}" --track="${track}" --cuda="${cuda}" --total-timesteps="${total_timesteps}" --num-envs="${num_envs}" --num-steps="${num_steps}" --num-minibatches="${num_minibatches}" --learning-rate="${learning_rate}" --ent-coef="${ent_coef}" --data-dir="${data_dir}"