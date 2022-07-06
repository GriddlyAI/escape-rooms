from escape_rooms.ppo import Agent
import torch

if __name__ == "__main__":

    agent_model = Agent((51, 9, 7), 12)

    # with open("agent_model_file.mdl", 'w') as f:
    torch.save({"model_state_dict": agent_model.state_dict()}, "agent_model.tar")
