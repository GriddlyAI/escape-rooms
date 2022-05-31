from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.level_generators.crafter_generator import CrafterLevelGenerator
from escape_rooms.level_generators.rotate_translate_generator import RotateTranslateGenerator

env = EscapeRoomWrapper(level_generator_cls=RotateTranslateGenerator)

env.reset(seed=567)

env.render(observer="global")

pass