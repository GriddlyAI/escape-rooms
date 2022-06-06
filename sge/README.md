# Grid Engine Param Sweeper

This folder contains tools for generating array jobs for Sun/Univa Grid Engine clusters.

Array jobs are very useful for queing hyperaparameter searches, large scale experiments with many different hyperparameters and settings.

Th ``param_sweeper.py`` script generates scripts for running ``train.py`` with many different parameter sets. 

## Generating arrays

You can create scripts using python and ``get_script`` in ``param_sweeper.py``:

Create a sweep across 3 different values of learning rate and entropy coefficient:
```python
job_name = 'my_job_name'
script = get_script(
        {
            'sge_time_h': 2,
            'sge_job_name': f'{job_name}',
            'sge_num_cpus': 8,
            'sge_num_gpus': 1,
            'sge_memory': 11,
            'sge_memory_unit': 'G',
            'sge_cluster_name': 'andrena',
            'sge_gpu_type': 'ampere',
            'sge_root_directory': '~/enn/incubator',
            'sge_entry_point': '~/enn/incubator/enn_ppo/enn_ppo/train.py'
        },
        {
            'gym-id': ['GDY-Clusters-0', 'GDY-Clusters-1', 'GDY-Clusters-2', 'GDY-Clusters-3', 'GDY-Clusters-4'],
            'exp-name': [f'{job_name}'],
            'track': ['True'],
            'total-timesteps': [50000000],
            'processes': [8],
            'num-envs': [128],
            'learning-rate': [0.005, 0.001, 0.0005],
            'ent-coef': [0.2, 0.1, 0.05],
            'eval-interval': [50000],
            'eval-steps': [300],
            'eval-num-env': [8],
            'eval-processes': [4],
            'data-dir': [f'/data/scratch/acw434/{job_name}']
        })
```

The created script will look something like this:

```shell
#!/bin/sh
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -N griddly-clusters-sge-sweep
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -l h_rt=2:0:0
#$ -t 1-44
#$ -o logs/
#$ -e logs/

gym_id_values=( GDY-Clusters-0 GDY-Clusters-1 GDY-Clusters-2 GDY-Clusters-3 GDY-Clusters-4 )
exp_name_values=( griddly-clusters-sge-sweep )
track_values=( True )
total_timesteps_values=( 50000000 )
processes_values=( 8 )
num_envs_values=( 128 )
learning_rate_values=( 0.005 0.001 0.0005 )
ent_coef_values=( 0.2 0.1 0.05 )
eval_interval_values=( 50000 )
eval_steps_values=( 300 )
eval_num_env_values=( 8 )
eval_processes_values=( 4 )
data_dir_values=( /data/scratch/acw434/griddly-clusters-sge-sweep )
trial=${SGE_TASK_ID}
gym_id=${gym_id_values[$(( trial % ${#gym_id_values[@]} ))]}
trial=$(( trial / ${#gym_id_values[@]} ))
exp_name=${exp_name_values[$(( trial % ${#exp_name_values[@]} ))]}
trial=$(( trial / ${#exp_name_values[@]} ))
track=${track_values[$(( trial % ${#track_values[@]} ))]}
trial=$(( trial / ${#track_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}
trial=$(( trial / ${#total_timesteps_values[@]} ))
processes=${processes_values[$(( trial % ${#processes_values[@]} ))]}
trial=$(( trial / ${#processes_values[@]} ))
num_envs=${num_envs_values[$(( trial % ${#num_envs_values[@]} ))]}
trial=$(( trial / ${#num_envs_values[@]} ))
learning_rate=${learning_rate_values[$(( trial % ${#learning_rate_values[@]} ))]}
trial=$(( trial / ${#learning_rate_values[@]} ))
ent_coef=${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}
trial=$(( trial / ${#ent_coef_values[@]} ))
eval_interval=${eval_interval_values[$(( trial % ${#eval_interval_values[@]} ))]}
trial=$(( trial / ${#eval_interval_values[@]} ))
eval_steps=${eval_steps_values[$(( trial % ${#eval_steps_values[@]} ))]}
trial=$(( trial / ${#eval_steps_values[@]} ))
eval_num_env=${eval_num_env_values[$(( trial % ${#eval_num_env_values[@]} ))]}
trial=$(( trial / ${#eval_num_env_values[@]} ))
eval_processes=${eval_processes_values[$(( trial % ${#eval_processes_values[@]} ))]}
trial=$(( trial / ${#eval_processes_values[@]} ))
data_dir=${data_dir_values[$(( trial % ${#data_dir_values[@]} ))]}

module purge
module load anaconda3 vulkan-sdk
conda activate poetry

export PYTHONUNBUFFERED=1

# Set up poetry
cd ~/enn/incubator
poetry shell

python ~/enn/incubator/enn_ppo/enn_ppo/train.py  --gym-id=${gym_id} --exp-name=${exp_name} --track=${track} --total-timesteps=${total_timesteps} --processes=${processes} --num-envs=${num_envs} --learning-rate=${learning_rate} --ent-coef=${ent_coef} --eval-interval=${eval_interval} --eval-steps=${eval_steps} --eval-num-env=${eval_num_env} --eval-processes=${eval_processes} --data-dir=${data_dir}
```

This can then be submitted using ``qsub`` which will run all the combinations