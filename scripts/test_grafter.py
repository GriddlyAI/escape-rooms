from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.play_wrapper import PlayWrapper
from escape_rooms.level_generators.crafter_generator import CrafterLevelGenerator

if __name__ == '__main__':
	# env = EscapeRoomWrapper(30, 30)
	env = EscapeRoomWrapper(player_observer_type='GlobalSprite2D', level_generator_cls=CrafterLevelGenerator)
	env = PlayWrapper(env, seed=100)
	env.play(fps=3)