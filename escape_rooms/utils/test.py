from escape_rooms.wrapper import EscapeRoomWrapper
from escape_rooms.level_generators.crafter_generator import CrafterLevelGenerator
from escape_rooms.level_generators.rotate_translate_generator import RotateTranslateGenerator

env = EscapeRoomWrapper(level_generator_cls=CrafterLevelGenerator)

env.reset()

env.render(observer="global")

pass