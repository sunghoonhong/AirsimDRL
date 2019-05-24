import time
import numpy as np
import airsim
import config

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5
floorZ = 1.18
goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0

    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation

    def step(self, quad_offset):
        # move with given velocity
        quad_offset = [float(i) for i in quad_offset]
        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.simPause(False)

        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], timeslice)
        # self.client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], timeslice)
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided
            # landed = quad_pos.y_val > 10 and self.client.getMultirotorState().landed_state == airsim.LandedState.Landed
            # landed = landed or (quad_pos.y_val > 10 and quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        self.client.simPause(True)

        # observe with depth camera
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])

        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        # decide whether done
        dead = has_collided or quad_pos.y_val <= outY
        done = dead or quad_pos.y_val >= goalY

        # compute reward
        reward = self.compute_reward(quad_pos, quad_vel, dead)

        # log info
        info = {}
        info['Y'] = quad_pos.y_val
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.y_val >= goalY:
            info['status'] = 'goal'
        else:
            info['status'] = 'going'
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation, reward, done, info

    def compute_reward(self, quad_pos, quad_vel, dead):
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        speed = np.linalg.norm(vel)
        if dead:
            reward = config.reward['dead']
        elif quad_pos.y_val >= goals[self.level]:
            self.level += 1
            # reward = config.reward['forward'] * (1 + self.level / len(goals))
            reward = config.reward['goal'] * (1 + self.level / len(goals))
        elif speed < speed_limit:
            reward = config.reward['slow']
        else:
            reward = float(vel[1]) * 0.1
        # elif vel[1] > 0:
        #     reward = config.reward['forward'] * (1 + self.level / len(goals))
        # else:
        #     reward = config.reward['normal']
        return reward
    
    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')