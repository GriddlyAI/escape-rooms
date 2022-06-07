# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from escape_rooms.level_generators.rotate_translate_generator import (
    LevelGenerator,
    RotateTranslateGenerator,
)
from escape_rooms.level_generators.crafter_generator import (
    CrafterLevelGenerator,
)
from escape_rooms.level_generators.human_generator import HumanDataGenerator
from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.procgen_wrapper import (
    UniformSeedSettingWrapper,
    SequentialSeedSettingWrapper,
)
from ppo import Agent

import _pickle as cPickle


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--checkpoint-dir", type=str, default="/private/home/samvelyan/grafter/Grafter_30x30__Grafter-Mon-1e-4-256-4-False__Grafter-Mon__1__1654534045")
    parser.add_argument("--model-tar", type=str, default="checkpoint_9000.tar")
    parser.add_argument("--levels", type=str, default="generator", choices=['generator', 'human'])
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Grafter args
    parser.add_argument("--height", type=int, default=30,
                        help="height of the grafter environments")
    parser.add_argument("--width", type=int, default=30,
                        help="width of the grafter environments")
    parser.add_argument("--observer-type", type=str, default="Vector",
                        help="Type of the observer")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    # fmt: on
    return args


def make_env(
    seed,
    idx,
    capture_video,
    run_name,
    level_generator_cls=CrafterLevelGenerator,
    max_seed=100,
):
    def thunk():
        # env = EscapeRoomWrapper(level_generator_cls=RotateTranslateGenerator)
        env = EscapeRoomWrapper(level_generator_cls=level_generator_cls)
        env = SequentialSeedSettingWrapper(env, max_seed=max_seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = args.checkpoint_dir.split("/")[-1]
    print(run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    max_seed = 99
    if args.levels == "generator":
        level_generator_cls = CrafterLevelGenerator
    else:
        level_generator_cls = HumanDataGenerator

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                level_generator_cls=level_generator_cls,
                max_seed=max_seed,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    observation_space = envs.single_observation_space
    action_space = envs.single_action_space

    # Load checkpointed agent
    agent = Agent(observation_space.shape, action_space.n).to(device)
    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    xs = [x for x in range(500, 6001, 500)]
    ys = []
    cp_tars = [f"checkpoint_{x}.tar" for x in xs]

    for x, cp_tar in zip(xs, cp_tars):

        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.seed + i,
                    i,
                    args.capture_video,
                    run_name,
                    level_generator_cls=level_generator_cls,
                    max_seed=max_seed,
                )
                for i in range(args.num_envs)
            ]
        )
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        checkpoint_path = os.path.join(args.checkpoint_dir, cp_tar)
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint["model_state_dict"])

        returns = []
        checkpoint_steps = 0
        global_step = 0
        episodes = 0
        while episodes < args.num_episodes:
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    device
                ), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        # print(
                        #     f"global_step={global_step}, episodic_return={item['episode']['r']}"
                        # )
                        returns.append(item["episode"]["r"])
                        episodes += 1

        print(
            f"Checkpoint {x}: Mean return = {np.mean(returns)}, steps - {global_step}, episodes = {episodes}"
        )
        ys.append(np.mean(returns))

    output_file_path = os.path.join(
        "eval_results", args.levels, run_name + ".pickle"
    )
    with open(output_file_path, "wb") as output_file:
        cPickle.dump(ys, output_file)
