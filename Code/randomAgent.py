import gym
import gym_gvgai
import random

# Get all of the GVGAI Environments
envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]

# Create and reset the enivronment
env = gym.make(random.choice(envs)) # gym.make('gvgai-aliens-lvl0-v0') <- Code to load a specific environment
env.reset()

# Set the score counter to 0
score = 0
# Run game for 2000 game ticks
for i in range(2000):
    # Render the game to the screen
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
        print("Game over at game tick " + str(i+1) + " with player " + info['winner'])
        break

# Close the environment render after game is finished
env.close()
