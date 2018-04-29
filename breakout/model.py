#import gym
import gym
import numpy as np




def to_grayscale(img):
    return np.mean(img,axis=2).astype(np.uint8)


def downsample(img):
    return img[::2,::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def transform_reward(reward):
    return np.sign(reward)









#create brakout environment
env = gym.make('BreakoutDeterministic-v4')
#reset it
frame = env.reset()
#render
env.render()

is_done = False
while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    #render
    env.render()