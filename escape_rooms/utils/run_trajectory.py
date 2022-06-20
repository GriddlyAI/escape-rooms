from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.level_generators.human_generator import HumanDataGenerator
import yaml
import os


def get_flat_action(action):
    """
    The trajectories are stored in Griddly's conditional action tree format, so we have to flatten them
    """
    offsets = {
        0: 0,
        1: 5,
        2: 6,
        3: 9
    }

    return offsets[action[0]] + action[1] - (1 if action[0]>0 else 0)

if __name__ == "__main__":
    # Set the level between 0 and 100 here
    level = 37

    env = EscapeRoomWrapper(level_generator_cls=HumanDataGenerator, player_observer_type="GlobalSprite2D")

    # Get the trajectory from the dataset
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "../trajectories/Grafter Escape Rooms.yaml"), 'r') as f:
        trajectories = yaml.load(f, yaml.SafeLoader)

        trajectory = trajectories[f"{level}"]
        env.seed(trajectory["seed"])
        env.reset(seed=level)
        for action in trajectory["steps"]:
            flat_action = get_flat_action(action)
            env.step(flat_action)

            env.render()
