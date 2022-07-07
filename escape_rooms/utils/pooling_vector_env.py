import multiprocessing as mp
import multiprocessing.connection as conn
from multiprocessing.connection import Connection

import gym.vector
import numpy as np
from gym.vector.utils import CloudpickleWrapper


def _worker(
        remote,
        parent_remote,
        env_config,
) -> None:
    parent_remote.close()
    env_cls = env_config.fn["env_cls"]
    num_envs = env_config.fn["num_envs"]
    observation_space = env_config.fn["observation_space"]
    action_space = env_config.fn["action_space"]

    envs_cls = [env_cls for _ in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(envs_cls, observation_space, action_space)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation = envs.step(data)
                remote.send(observation)
            elif cmd == "reset":
                observation = envs.reset(**data)
                remote.send(observation)
            elif cmd == "render":
                rgb_pixels = envs.render(**data)
                remote.send(rgb_pixels)
            elif cmd == "close":
                envs.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class PoolingVectorEnv(gym.vector.VectorEnv):
    """
    We fork the subprocessing from the stable-baselines implementation, but use RaggedBuffers for collecting batches

    Citation here: https://github.com/DLR-RM/stable-baselines3/blob/master/CITATION.bib
    """

    def __init__(
            self,
            env_cls,
            num_envs,
            num_processes,
            observation_space=None,
            action_space=None
    ):
        dummy_env = env_cls()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env

        super().__init__(num_envs, observation_space, action_space)
        assert (
                num_envs % num_processes == 0
        ), "The required number of environments can not be equally split into the number of specified processes."

        self.num_processes = num_processes
        self.envs_per_process = int(num_envs / num_processes)

        env_list_configs = [
            {
                "env_cls": env_cls,
                "num_envs": self.envs_per_process,
                "observation_space": observation_space,
                "action_space": action_space
            }
            for _ in range(self.num_processes)
        ]

        ctx = mp.get_context("spawn")

        self.remotes = []
        self.work_remotes = []
        for i in range(self.num_processes):
            pipe = ctx.Pipe()
            self.remotes.append(pipe[0])
            self.work_remotes.append(pipe[1])

        self.processes = []
        for work_remote, remote, env_list_config in zip(
                self.work_remotes, self.remotes, env_list_configs
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_list_config))
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(("reset", kwargs))

        # Empty initialized observation batch
        observations = []

        for remote in self.remotes:
            remote_obs_batch = remote.recv()
            observations.extend(remote_obs_batch)

        return np.stack(observations)

    def render(self, **kwargs):
        rgb_arrays = []
        for remote in self.remotes:
            remote.send(("render", kwargs))
            rgb_arrays.append(remote.recv())

        np_rgb_arrays = np.concatenate(rgb_arrays)
        assert isinstance(np_rgb_arrays, np.ndarray)
        return np_rgb_arrays

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

    def _chunk_actions(self, actions):
        for i in range(0, self.num_envs, self.envs_per_process):
            yield actions[i: i + self.envs_per_process]



    def step(self, actions):
        remote_actions = self._chunk_actions(actions)
        for remote, action in zip(self.remotes, remote_actions):
            remote.send(("step", (action)))

        # Empty initialized observation batch
        obs = []
        reward = []
        done = []
        info = []
        for remote in self.remotes:
            obs_i, reward_i, done_i, info_i = remote.recv()
            obs.extend(obs_i)
            reward.extend(reward_i)
            done.extend(done_i)
            info.extend(info_i)
        return np.stack(obs), np.stack(reward), np.stack(done), np.stack(info)

    def __len__(self) -> int:
        return self.num_envs
