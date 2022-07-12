# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch

from escape_rooms.level_generators.crafter_generator import (
    CrafterLevelGenerator,
)
from escape_rooms.level_generators.human_generator import HumanDataGenerator
from escape_rooms.utils.pooling_vector_env import PoolingVectorEnv
from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.procgen_wrapper import (
    SeedListWrapper,
)
from escape_rooms.ppo import Agent

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
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-processes", type=int, default=8,
                        help="the number of processes to spread the environments across")
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
    capture_video,
    run_name,
    level_generator_cls=CrafterLevelGenerator,
):
    def thunk(idx):
        # env = EscapeRoomWrapper(level_generator_cls=RotateTranslateGenerator)

        range_start = int((idx * max_seed) / args.num_envs)
        range_end = int(((idx + 1) * max_seed) / args.num_envs)
        seeds = np.arange(range_start, range_end).tolist()

        env = EscapeRoomWrapper(level_generator_cls=level_generator_cls)
        env = SeedListWrapper(env, seeds=seeds)
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

    max_seed = 100
    if args.levels == "generator":
        level_generator_cls = CrafterLevelGenerator
    else:
        level_generator_cls = HumanDataGenerator


    # env setup
    envs = PoolingVectorEnv(
        make_env(args.seed, args.capture_video, run_name, level_generator_cls=level_generator_cls),
        1,
        1,
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

    xs = [x for x in range(0, 1575, 75)]
    ys = []
    achievements = []
    cp_tars = [f"checkpoint_{1 if x == 0 else x }.tar" for x in xs]

    envs.close()

    for x, cp_tar in zip(xs, cp_tars):

        # env setup
        envs = PoolingVectorEnv(
            make_env(args.seed, args.capture_video, run_name, level_generator_cls=level_generator_cls),
            args.num_envs,
            args.num_processes,
        )

        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        checkpoint_path = os.path.join(args.checkpoint_dir, cp_tar)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        agent.load_state_dict(checkpoint["model_state_dict"])

        returns = []
        ach_eat_plants = []
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
                        if 'ignore' not in item.keys():
                            episodes += 1
                            print(
                                f"global_step={global_step}, episodic_return={item['episode']['r']}, episodes: {episodes}"
                            )
                            returns.append(item["episode"]["r"])
                            ach_eat_plants.append(item["ach_eat_plant"])


        print(
            f"Checkpoint {x}: Mean return = {np.mean(returns)}, eaten = {np.mean(ach_eat_plants)}, steps - {global_step}, episodes = {episodes}"
        )
        ys.append(np.mean(returns))
        achievements.append(np.mean(np.mean(ach_eat_plants)))
        envs.close()

    os.makedirs(os.path.join("eval_results", args.levels), exist_ok=True)
    output_file_path = os.path.join(
        "eval_results", args.levels, run_name + ".pickle"
    )
    with open(output_file_path, "wb") as output_file:
        cPickle.dump((ys, achievements), output_file)
