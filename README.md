# PySTK2-gymnasium / BBRL project template

This project template contains a basic structure that could be used for your PySTK2/BBRL project.
For information about the PySTK2 gymnasium environment, please look at the [corresponding github page](https://github.com/bpiwowar/pystk2-gymnasium)

## Structure

**Warning**: all the imports should be relative within your module (see `learn.py` for an example).

### `actors.py`

Contains the actors used throughout your project

### `learn.py`

Contains the code to train your actor

### `pystk_actor.py`

This Python file (**don't change its name**) should contain:

- `env_name`: The base environment name
- `player_name`: The actor name (displayed on top of the kart)
- `get_actor(state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space)`. It should return an actor that writes into `action` or `action/...`. It should *not* return a temporal agent. The parameters of the agent should be saved with `torch.save(actor.state_dict(), "pystk_actor.pth")`



### Learn your model

```sh
# To be run from the base directory
PYTHONPATH=. python -m stk_actor.learn
```

This should create the `pystk_actor.pth` file (**don't change its name**) that contains the parameters of your model. The file will be loaded using `torch.load(...)` and the data will be transmitted as  a parameter to `get_actor` (see `pystk_actor.py`).


# Testing the actor

You can use [master-mind](https://pypi.org/project/su_master_mind/) to test your agent (you can even experiment with races between different actors to select the one of your choice):

To test your agent, you can also use the module name (so you can compare different actors in the same project)
```sh
# Replace stk_actor by something else if testing different actors
PYTHONPATH=. master-mind rl stk-race --hide stk_actor
```

# 🧭 Submit your work on the `evaluation` branch

1. **Commit your final work**

   ```bash
   # Add files
   git add ...
   git commit -m "Final version for evaluation"
   ```

   Ensure that all relevant files are versioned with `git status` – in particular the `pystk_actor.pth`.

2. **Switch (or create) the `evaluation` branch**

   ```bash
   git fetch origin
   git checkout -B evaluation origin/evaluation || git checkout -b evaluation
   ```


3. **Merge your work and push**

   ```bash
   git merge main        # or your working branch
   git push origin evaluation
   ```

   In case of problem with the `pystk_actor.pth` file, use this
   ```bash
    # overwrite binary file
    git checkout main -- stk_actor/pystk_actor.pth
    git add stk_actor/pystk_actor.pth
   ```

4. ✅ **Check online** that the `evaluation` branch contains your latest commit.
