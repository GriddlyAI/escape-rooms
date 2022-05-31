from wrappers import EscapeRoomWrapper, PlayWrapper

if __name__ == '__main__':
	env = EscapeRoomWrapper(30, 30)
	env = PlayWrapper(env, seed=100)
	env.play(fps=3)