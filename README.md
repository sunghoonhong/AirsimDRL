# Deep Reinforcement Learning for Airsim Environment
Quadrotor Self-Flight using Depth image

#### NOTE
It is a capstone project for undergraduate course.
It did work when I tried, but there were many trial and errors.
I'm sorry that I didn't consider any reproducibility (e.g. random seed).

##### Check 1 min madness
[![1 min madness video](/images/demo.gif)](https://youtu.be/C9P0V5Hif54)
# Environment

## Link to download executable
#### NOTE: These executables can be run only on Windows OS.
[Easy](https://drive.google.com/file/d/1LigXGvDj0XZvgkffqBwe8XRWRmzMR93P/view?usp=sharing)
[Normal](https://drive.google.com/file/d/1KtiHr_qpw37qq3PPiAKzLN9THm2aQZOU/view?usp=sharing)
[Hard](https://drive.google.com/file/d/110mekUMdnYr5wNaEGVbsSZpwty12knzX/view?usp=sharing)


## How To Use
Execute the environment first.
If you can see the rendered simulation, then run what you want to try (e.g. python td3_per.py)

## Description
Unreal Engine 4

- Original environment
1. Vertical column
2. Horizontal column
3. Window
4. Vertical curved wall


<p align="center">
    <img src="/images/1.png" width="280" height="200">
    <img src="/images/2.png" width="280" height="200">
    <img src="/images/3.png" width="280" height="200">
</p>

- Different Order of obstacles environment
1. Window
2. Horizontal column
3. Vertical curved wall
4. Vertical column


- Different type of obstacles environment
1. Horizontal curved wall
2. Reversed ㄷ shape
3. ㄷ shape
4. Diagonal column


<p align="center">
    <img src="/images/6.jpg" width="280" height="200">
    <img src="/images/5.png" width="280" height="200">
    <img src="/images/4.jpg" width="280" height="200">
</p>

### Parameter
- Timescale: 0.5 (Unit time for each step)
- Clockspeed: 1.0 (Default)
- Goals: [7, 17, 27.5, 45, 57]
- Start position: (0, 0, 1.2)

### Reset
Respawn at the start position, and then take off and hover.  
It takes about 1 sec.

### Step
Given action as 3 real value, process *moveByVelocity()* for 0.5 sec.  
For delay caused by computing network, pause Simulation after 0.5 sec.

### Done
If a collision occurs, including landing, it would be dead.
If x coordinate value is smaller than -0.5, it would be dead.
If it gets to the final goal, the episode would be done.

### State
- Depth images from front camera (144 \* 256 or 72 \* 128)
- (Optional) Linear velocity of quadrotor (x, y, z)

### Action
- Discrete Action Space (Action size = 7)  
Using *interpret_action()*, choose +/-1 along one axis among x, y, z or hovering.


- Continuous Action Space (Actions size = 3)  
3 real values for each axis. I decided the scale as 1.5 and gave a bonus for y axis +0.5.

### Reward
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

# Result
<img src="/save_graph/result_Best Record.png" height="200">
<img src="/save_graph/result_Get Goal Prob..png" height="200">
