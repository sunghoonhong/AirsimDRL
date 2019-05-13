# Deep Reinforcement Learning for Airsim Environment
Quadrotor Self-Flight using Depth image

# Environment

## Parameter
- Timescale: 0.5 (Unit time for each step)
- Clockspeed: 1.0 (Default)
- Goals: [7, 17, 27.5, 45, 57]
- Start position: (0, 0, 1.2)

## Reset
Respawn at the start position, and then take off and hover.  
It takes about 1 sec.

## Step
Given action as 3 real value, process *moveByVelocity()* for 0.5 sec.  
For delay caused by computing network, pause Simulation after 0.5 sed.

## Done
If a collision occurs, including landing, it would be dead.
If x coordinate value is smaller than -0.5, it would be dead.
If it gets to the final goal, the episode would be done.

## State
- Depth images from front camera (144 \* 256 or 72 \* 128)
- (Optional) Linear velocity of quadrotor (x, y, z)

## Action
- Discrete Action Space (Action size = 7)  
Using *interpret_action()*, choose +/-1 along one axis among x, y, z or hovering.


- Continuous Action Space (Actions size = 3)  
3 real values for each axis. I decided the scale as 1.5 and gave a bonus for y axis +0.5.

## Reward
- Dead: -2.0
- Goal: 2.0 * (1 + level / # of total levels)
- Too slow(Speed < 0.2): -0.05
- Otherwise: 0.1 * linear velocity along y axis  

(e.g. The faster go forward, The more reward is given. The faster go backward, The more penalty is given.)

# Agent
- Recurrent DQN
- Recurrent A2C
- Recurrent DDPG
- Recurrent DDPG + PER
- __Recurrent TD3 + PER (BEST)__
