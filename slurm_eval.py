import datetime
import submitit
import os
import sys
from coolname import generate_slug

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "~/cmds.txt", "Path to list of commands to run.")
flags.DEFINE_string("name", "anyslurm", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Only debugging output.")


def arg2str(k, v):
    if isinstance(v, bool):
        if v:
            return ("--%s" % k,)
        else:
            return ""
    else:
        return ("--%s" % k, str(v))


class LaunchExperiments:
    def __init__(self):
        pass

    def launch_experiment_and_remotenv(self, experiment_args):
        # imports and definition are inside of function because of submitit
        import multiprocessing as mp

        def launch_experiment(experiment_args):
            import subprocess

            python_exec = sys.executable

            subprocess.call([python_exec, "-m", "eval"] + experiment_args)

        experiment_process = mp.Process(
            target=launch_experiment, args=[experiment_args]
        )
        self.process = experiment_process
        experiment_process.start()
        experiment_process.join()

    def __call__(self, experiment_args):
        self.launch_experiment_and_remotenv(experiment_args)


def main(argv):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/slurm/{FLAGS.name}")
    submitit_dir = os.path.expanduser(f"~/slurm/{FLAGS.name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir, max_num_timeout=5)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    with open(os.path.expanduser(FLAGS.path), "r") as f:
        cmds = "".join(f.readlines()).split("\n")
        cmds = [cmd.split()[2:] for cmd in cmds if len(cmd) > 0]

    executor.update_parameters(
        # examples setup
        partition="learnlab",
        comment="NeurIPS 2022 submission",
        time=1 * 72 * 60,
        nodes=1,
        ntasks_per_node=1,
        mem="100GB",
        # job setup
        job_name=FLAGS.name,
        cpus_per_task=10,
        gpus_per_node=1,
        array_parallelism=256,
    )

    print("\nAbout to submit", len(cmds), "jobs")

    if not FLAGS.debug:
        job = executor.map_array(LaunchExperiments(), cmds)

        for j in job:
            print("Submitted with job id: ", j.job_id)
            print(f"stdout -> {submitit_dir}/{j.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{j.job_id}_0_log.err")

        print(f"Submitted {len(job)} jobs!")

        print()
        print(submitit_dir)


if __name__ == "__main__":
    app.run(main)
