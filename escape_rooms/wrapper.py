from pathlib import Path

import gym
from griddly import GymWrapper
from griddly.util.action_space import MultiAgentActionSpace

from escape_rooms.level_generators.crafter_generator import CrafterLevelGenerator


class EscapeRoomWrapper(gym.Wrapper):
    def __init__(
            self,
            player_observer_type="PlayerSprite2D",
            global_observer_type="GlobalSprite2D",
            level_generator=None,
    ):

        current_file = Path(__file__).parent

        self._genv = GymWrapper(
            yaml_file=str(current_file.joinpath("gdy").joinpath("grafter-escape-rooms.yaml")),
            global_observer_type=global_observer_type,
            player_observer_type=player_observer_type,
            gdy_path=str(current_file.joinpath("gdy")),
            image_path=str(current_file.joinpath("assets")),
        )

        if level_generator is not None:
            self._generator = level_generator

        self._genv.reset()

        super().__init__(self._genv)

        # flatten the action space
        self.action_space, self.flat_action_mapping = self._flatten_action_space()

    def _flatten_action_space(self):
        flat_action_mapping = []
        actions = []
        actions.append("NOP")
        flat_action_mapping.append([0, 0])
        for action_type_id, action_name in enumerate(self._genv.action_names):
            action_mapping = self._genv.action_input_mappings[action_name]
            input_mappings = action_mapping["InputMappings"]

            for action_id in range(1, len(input_mappings) + 1):
                mapping = input_mappings[str(action_id)]
                description = mapping["Description"]
                actions.append(description)

                flat_action_mapping.append([action_type_id, action_id])

        action_space = gym.spaces.Discrete(len(flat_action_mapping))

        return action_space, flat_action_mapping

    def step(self, action):
        g_action = self.flat_action_mapping[action]
        return self.env.step(g_action)

    def reset(self):
        if self._generator is not None:
            level_string = self._generator.generate()
            reset_obs = self.env.reset(level_string=level_string)
        else:
            reset_obs = self.env.reset(level_id=self._level_id)

        return reset_obs

    def render(self, mode="human", observer=0):
        return self.env.render(mode=mode, observer=observer)
