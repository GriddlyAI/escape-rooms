# Grafter Escape Rooms

Dataset and experiment details for Escape Rooms built using [GriddlyJS](griddly.ai).

![Escape Rooms](figures/escape_rooms.png)


## Installation

clone this repository and install the dependencies

```shell
pip install -r requirements.txt
```

## Human Trajectories

We provide recorded trajectories for each of the 100 levels [here](escape_rooms/trajectories/Grafter%20Escape%20Rooms.yaml).

### Example Usage

The trajectories dataset can be loaded and run using [run_trajectory.py](escape_rooms/utils/run_trajectory.py)

which looks like this:
```python
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
```

Simply set the `level` variable between 0 and 100 to load the trajectory for each level.


## Training

To train using the PCG generated dataset you can run the following command:

```shell
~/escape-rooms/ppo.py  --wandb-entity="my_wandb_name" --exp-name="griddly-escape-rooms" --track="true" --cuda="true" --total-timesteps="10000000" --num-envs="64" --num-steps="512" --learning-rate="0.005" --ent-coef="0.1" --seed="10" --checkpoint-path="checkpoints" --checkpoint-interval="30"
```

## Evaluation

To evaluate a trained set of checkpoints against the human designed escape rooms:

```shell
python eval.py --levels human --seed 48 --checkpoint-dir checkpoints/Grafter_30x30__Grafter-Mon-1e-4-256-4-False__Grafter-Mon__1__1654534045
```



