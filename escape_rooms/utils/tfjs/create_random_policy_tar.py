from ppo import Agent
import torch

if __name__ == "__main__":

    agent_model = Agent((7, 9, 51), 12)

    # with open("agent_model_file.mdl", 'w') as f:
    torch.save({"model_state_dict": agent_model.state_dict()}, "agent_model.tar")
