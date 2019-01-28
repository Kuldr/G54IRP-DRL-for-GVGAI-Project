import gym
import gym_gvgai

from stable_baselines import A2C

env = gym.make('gvgai-boulderdash-lvl0-v0')
model = A2C.load("models/a2c_boulderdash_1M")

# Set the score counter to 0
score = 0
# Reset the enivronment
stateObs = env.reset()
# Reset the time counter t
t = 0

while True:
    actionID, _ = model.predict(stateObs)

    # Perform the action choosen and get the info from the environment
    stateObs, reward, isOver, info = env.step(actionID)
    # Update the cumilative score based upon the reward given
    score += reward

    # Print the results of the action performed
    print("Action " + str(actionID) + " played at game tick " + str(t+1) + ", reward=" + str(reward) + ", new score=" + str(score))
    if isOver:
        print("Game over at game tick " + str(t+1) + " with player " + info['winner'] + ". Score = " + str(score) + " at Env " + str(env))
        score = 0
        stateObs = env.reset()
        t = 0

    env.render()
    t += 1
