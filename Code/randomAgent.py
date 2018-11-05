import gym
import gym_gvgai
import random

RENDER_TO_SCREEN = False
GAME_TICKS = 2000

# Get all of the GVGAI Environments
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]

# Create and reset the enivronment
env = gym.make(random.choice(envs)) # gym.make('gvgai-aliens-lvl0-v0') <- Code to load a specific environment
env.reset()

# Print the current enivronment
print(env)

# Set the score counter to 0
score = 0
# Run game for a number game ticks
for i in range(GAME_TICKS):
    # Render the game to the screen if option is selected
    if RENDER_TO_SCREEN:
        env.render()
    # Select a random action from the action space
    action_id = env.action_space.sample()
    # Perform the action choosen and get the info from the environment
    state, reward, isOver, info = env.step(action_id)
    # Update the cumilative score based upon the reward given
    score += reward

    # Print the results of the action performed
    print("Action " + str(action_id) + " played at game tick " + str(i+1) + ", reward=" + str(reward) + ", new score=" + str(score))
    if isOver:
        print("Game over at game tick " + str(i+1) + " with player " + info['winner'] + ". Score = " + str(score) + " at Env " + str(env))
        break

# Close the environment render after game is finished
if RENDER_TO_SCREEN:
    env.close()
