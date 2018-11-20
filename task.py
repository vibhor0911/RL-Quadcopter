import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        total_effort_up = (self.target_pos[2] - self.sim.pose[2])
        reward = 0
        if total_effort_up >= 0:
            reward = 1
        else:
            reward = -1
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
class TakeOff():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a TakeOff object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        z_up = (self.target_pos[2] - self.sim.pose[2])
        z = (self.target_pos[2])
        y_up = (self.target_pos[1] - self.sim.pose[1])
        y = (self.target_pos[1])
        x_up = (self.target_pos[0] - self.sim.pose[0])
        x = (self.target_pos[0])
        #deviation = self.sim.pose[1] + self.sim.pose[0]        
        reward = 0
        if (z_up == 0 and self.sim.pose[1] == 0 and self.sim.pose[0]==0):
            reward = 10**10
        elif (z_up ==0 and self.sim.pose[1] != 0 and self.sim.pose[0]!=0):
            reward = 10 **8
        elif abs(z_up) > z:
            reward = -10**26 * (abs(z_up))
        elif self.sim.pose[1] != 0 and self.sim.pose[0]!=0:
            reward = 10**20/((abs(z_up * 3)**2 + abs(y_up * 1.2)**2 + abs(x_up)**2))
        else:
            reward = 10**20/(abs(z_up)**2)
        #reward = reward/((abs(z_up)**2 + abs(y_up)**2 + abs(x_up)**2)**0.5)
        return reward 

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    
    
class Landing():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a TakeOff object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.init_pose = init_pose
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        z_up = (self.target_pos[2] - self.sim.pose[2])
        
        y_up = (self.target_pos[1] - self.sim.pose[1])
        
        x_up = (self.target_pos[0] - self.sim.pose[0])
        if abs(self.sim.pose[2]) > self.init_pose[2]:
            reward = -(100)  * abs(self.sim.pose[2])-abs(self.init_pose[2])            
        elif self.sim.pose[2] < -100:
            reward = -(100) * self.sim.pose[2]
        elif self.sim.pose[1] == 0 and self.sim.pose[0] == 0:
            if self.sim.pose[2] == 0:
                reward = 10**5
            elif z_up!=0:
                reward = (10**2) / abs(z_up)
        elif abs(z_up)+abs(y_up)+abs(x_up)!=0:
            reward = 10**2/(abs(z_up * 2)**2 + abs(y_up * 1.3)**2 + abs(x_up)**2)
        elif abs(z_up)+abs(y_up)+abs(x_up)==0:
            reward = 10**5
        else:
            reward = 0
        return reward
                             
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    