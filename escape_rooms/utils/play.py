from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.level_generators.human_generator import HumanDataGenerator
from escape_rooms.play_wrapper import PlayWrapper

if __name__ == "__main__":

    # Set the level between 0 and 100 here
    level=0

    env = EscapeRoomWrapper(level_generator_cls=HumanDataGenerator, player_observer_type="PlayerSprite2D")

    env = PlayWrapper(env, seed=level)

    env.play(fps=10)

