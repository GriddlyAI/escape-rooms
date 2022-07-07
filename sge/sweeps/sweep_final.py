from sge.param_sweeper import get_script

if __name__ == '__main__':
    job_name = 'escape-rooms-final-July-50M-gae'
    script = get_script(
        {
            'sge_time_h': 10,
            'sge_job_name': f'{job_name}',
            'sge_num_cpus': 8,
            'sge_num_gpus': 1,
            'sge_memory': 11,
            'sge_memory_unit': 'G',
            #'sge_cluster_name': 'andrena',
            'sge_gpu_type': 'ampere',
            'sge_root_directory': '~/escape-rooms',
            'sge_entry_point': '~/escape-rooms/escape_rooms/ppo.py'
        },
        {
            'wandb-entity': ['chrisbam4d'],
            'exp-name': [f'{job_name}'],
            'track': ['True'],
            'cuda': ['True'],
            'total-timesteps': [50000000],
            'num-envs': [64],
            'num-steps': [512],
            'learning-rate': [0.001],
            'ent-coef': [0.01],
            'gae-lambda': [0.65, 0.8],
            'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'data-dir': [f'/data/scratch/acw434/{job_name}'],
            'checkpoint-path': [f'/data/scratch/acw434/{job_name}/checkpoints'],
            'checkpoint-interval': [6250]
        })

    with open(f'submit-array_{job_name}.sh', 'w') as f:
        f.write(script)
