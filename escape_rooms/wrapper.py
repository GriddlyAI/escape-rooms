from pathlib import Path
import numpy as np

import yaml
import gym
from griddly import GymWrapper
from griddly import gd


class EscapeRoomWrapper(gym.Wrapper):
    def __init__(
            self,
            player_observer_type="Vector",
            global_observer_type="GlobalSprite2D",
            level_generator_cls=None,
    ):
        current_file = Path(__file__).parent
        with open(current_file.joinpath("gdy").joinpath("grafter-escape-rooms.yaml")) as f:
            gdy = yaml.safe_load(f)

            self._level_generator = level_generator_cls(gdy)

            yaml_string = yaml.dump(gdy)

        self._genv = GymWrapper(
            yaml_string=yaml_string,
            yaml_file=str(current_file.joinpath("gdy").joinpath("grafter-escape-rooms.yaml")),
            global_observer_type=global_observer_type,
            player_observer_type=player_observer_type,
            gdy_path=str(current_file.joinpath("gdy")),
            image_path=str(current_file.joinpath("assets")),
        )

        self._vector_global_vars = ["inv_wood_pickaxe", "inv_stone_pickaxe", "inv_iron_pickaxe", "inv_stone",
                                    "inv_coal", "inv_wood", "inv_iron", "health"]
        self._num_global_vars = len(self._vector_global_vars)
        self._genv.reset()

        super().__init__(self._genv)

        # flatten the action space
        self.action_space, self.flat_action_mapping = self._flatten_action_space()
        self.observation_space = self._calc_observation_space(self._num_global_vars)

    def _calc_observation_space(self, num_global_vars):
        if self.env._player_observer_type[0] == gd.ObserverType.VECTOR:
            space_with_global_vars = [
                self.observation_space.shape[0] + num_global_vars,
                self.observation_space.shape[1],
                self.observation_space.shape[2]
            ]

            return gym.spaces.Box(low=0, high=1, shape=space_with_global_vars, dtype=np.uint8)

    def _flatten_action_space(self):
        flat_action_mapping = []
        actions = []
        actions.append("NOP")
        flat_action_mapping.append([0, 0])
        for action_type_id, action_name in enumerate(self._genv.action_names):
            action_mapping = self.env.action_input_mappings[action_name]
            input_mappings = action_mapping["InputMappings"]

            for action_id in range(1, len(input_mappings) + 1):
                mapping = input_mappings[str(action_id)]
                description = mapping["Description"]
                actions.append(description)

                flat_action_mapping.append([action_type_id, action_id])

        action_space = gym.spaces.Discrete(len(flat_action_mapping))

        return action_space, flat_action_mapping

    def _wrap_vector_obs(self, obs):
        global_vars = [g[1] for k, g in self.env.game.get_global_variable(self._vector_global_vars).items()]
        global_var_values = np.array(global_vars, dtype=np.uint8)
        tiled_global_vars = np.tile(global_var_values.reshape((8, 1, 1)), reps=(1, obs.shape[1], obs.shape[2]))
        return np.concatenate([obs, tiled_global_vars], axis=0)

    def step(self, action):
        g_action = self.flat_action_mapping[action]

        if self.env._player_observer_type[0] == gd.ObserverType.VECTOR:
            # Sellotape the global variable we care about to the obs
            obs, reward, info, done = self.env.step(g_action)
            all_obs = self._wrap_vector_obs(obs)
        else:
            all_obs, reward, info, done = self.env.step(g_action)

        return all_obs, reward, info, done

    def reset(self, seed=100):
        level_string = self._level_generator.generate(seed)
        obs = self.env.reset(level_string=level_string)
        self.observation_space = self._calc_observation_space(self._num_global_vars)
        return self._wrap_vector_obs(obs)

    def render(self, mode="human", observer=0):
        return self.env.render(mode=mode, observer=observer)
