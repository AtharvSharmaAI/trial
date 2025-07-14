# Simple Pygame + Reinforcement Learning Car Driving Environment with PyTorch RL Agent
# Part 1: Pygame environment + Custom RL wrapper + PyTorch DQN agent (No gym dependency)

import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize pygame
pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CAR_WIDTH, CAR_HEIGHT = 40, 70

class CarEnv:
    def __init__(self):
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RL Car Simulation")

        self.car = pygame.Rect(100, 500, CAR_WIDTH, CAR_HEIGHT)
        self.clock = pygame.time.Clock()

        self.state_dim = 5
        self.action_dim = 4  # 0 = forward, 1 = left, 2 = right, 3 = brake

        self.traffic_light_state = 0
        self.zebra_y = 250
        self.timer = 0
        self.reset()

    def step(self, action):
        reward = 0
        done = False

        if action == 0:
            self.speed = min(self.speed + 0.2, self.max_speed)
        elif action == 1:
            self.car.x -= 5
        elif action == 2:
            self.car.x += 5
        elif action == 3:
            self.speed = max(self.speed - 0.5, 0)

        self.car.y -= int(self.speed)

        self.timer += 1
        if self.timer % 120 == 0:
            self.traffic_light_state = 1 - self.traffic_light_state

        if self.car.y < self.zebra_y and self.traffic_light_state == 0:
            reward -= 10
        elif self.car.y < self.zebra_y and self.traffic_light_state == 1:
            reward += 2
        else:
            reward += 0.1

        if self.car.x < 0 or self.car.x > SCREEN_WIDTH - CAR_WIDTH:
            reward -= 100
            done = True

        if self.car.y < 0:
            done = True

        obs = self._get_obs()
        return obs, reward, done

    def reset(self):
        self.car.x = 100
        self.car.y = 500
        self.speed = 0
        self.max_speed = 5
        self.timer = 0
        self.traffic_light_state = 0
        return self._get_obs()

    def _get_obs(self):
        dist_to_zebra = max(0, self.car.y - self.zebra_y)
        return np.array([self.car.x, self.car.y, self.speed, self.traffic_light_state, dist_to_zebra], dtype=np.float32)

    def render(self):
        self.display.fill((50, 50, 50))
        pygame.draw.rect(self.display, (255, 255, 255), (300, 0, 200, SCREEN_HEIGHT))
        pygame.draw.line(self.display, (255, 255, 255), (0, self.zebra_y), (SCREEN_WIDTH, self.zebra_y), 5)
        pygame.draw.rect(self.display, (255, 0, 0), self.car)
        color = (255, 0, 0) if self.traffic_light_state == 0 else (0, 255, 0)
        pygame.draw.circle(self.display, color, (700, self.zebra_y - 50), 15)
        pygame.display.update()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

# Training loop
if __name__ == '__main__':
    env = CarEnv()
    input_dim = env.state_dim
    output_dim = env.action_dim

    dqn = DQN(input_dim, output_dim)
    target_dqn = DQN(input_dim, output_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer()

    episodes = 200
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()
            if random.random() < epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = dqn(torch.FloatTensor(state).unsqueeze(0))
                    action = q_vals.argmax().item()

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = dqn(states).gather(1, actions)
                next_q_values = target_dqn(next_states).max(1)[0].detach().unsqueeze(1)
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    env.close()
