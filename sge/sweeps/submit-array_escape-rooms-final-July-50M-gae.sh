#!/bin/sh
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -N escape-rooms-final-July-50M-gae
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l h_rt=10:0:0
#$ -t 1-20
#$ -o logs/
#$ -e logs/

wandb_entity_values=( chrisbam4d )
exp_name_values=( escape-rooms-final-July-50M-gae )
track_values=( True )
cuda_values=( True )
total_timesteps_values=( 50000000 )
num_envs_values=( 64 )
num_steps_values=( 512 )
learning_rate_values=( 0.001 )
ent_coef_values=( 0.01 )
gae_lambda_values=( 0.65 0.8 )
seed_values=( 0 1 2 3 4 5 6 7 8 9 )
data_dir_values=( /data/scratch/acw434/escape-rooms-final-July-50M-gae )
checkpoint_path_values=( /data/scratch/acw434/escape-rooms-final-July-50M-gae/checkpoints )
checkpoint_interval_values=( 6250 )
trial=${SGE_TASK_ID}
wandb_entity="${wandb_entity_values[$(( trial % ${#wandb_entity_values[@]} ))]}"
trial=$(( trial / ${#wandb_entity_values[@]} ))
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
learning_rate="${learning_rate_values[$(( trial % ${#learning_rate_values[@]} ))]}"
trial=$(( trial / ${#learning_rate_values[@]} ))
ent_coef="${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}"
trial=$(( trial / ${#ent_coef_values[@]} ))
gae_lambda="${gae_lambda_values[$(( trial % ${#gae_lambda_values[@]} ))]}"
trial=$(( trial / ${#gae_lambda_values[@]} ))
seed="${seed_values[$(( trial % ${#seed_values[@]} ))]}"
trial=$(( trial / ${#seed_values[@]} ))
data_dir="${data_dir_values[$(( trial % ${#data_dir_values[@]} ))]}"
trial=$(( trial / ${#data_dir_values[@]} ))
checkpoint_path="${checkpoint_path_values[$(( trial % ${#checkpoint_path_values[@]} ))]}"
trial=$(( trial / ${#checkpoint_path_values[@]} ))
checkpoint_interval="${checkpoint_interval_values[$(( trial % ${#checkpoint_interval_values[@]} ))]}"

module purge
module load cuda anaconda3 vulkan-sdk
conda activate escape

export PYTHONUNBUFFERED=1

cd ~/escape-rooms

python ~/escape-rooms/escape_rooms/ppo.py  --wandb-entity="${wandb_entity}" --exp-name="${exp_name}" --track="${track}" --cuda="${cuda}" --total-timesteps="${total_timesteps}" --num-envs="${num_envs}" --num-steps="${num_steps}" --learning-rate="${learning_rate}" --ent-coef="${ent_coef}" --gae-lambda="${gae_lambda}" --seed="${seed}" --data-dir="${data_dir}" --checkpoint-path="${checkpoint_path}" --checkpoint-interval="${checkpoint_interval}"