import math
import random
from collections import deque, namedtuple

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

b_size = 128
GAMMA = 0.99


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    steps_done = 0

    def __init__(self, input_size, output_size, lr=1e-4, gamma=0.99, epsilon_start=1, epsilon_end=0.05,
                 epsilon_decay=1000, tau=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1. * Agent.steps_done / self.epsilon_decay)
        Agent.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < b_size:
            return

        transitions = self.memory.sample(b_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # current Q

        next_state_values = torch.zeros(b_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # r + dis * Q(s')

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


def plot_learning_curve(scores):
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig('Acrobot_score.png')
    plt.show()


def main():
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    agent = Agent(input_size, output_size)
    print(agent.device)
    num_episodes = 500
    episode_scores = []

    with open('results.txt', 'w') as results_file:
        for episode in range(num_episodes):
            state, _ = env.reset()
            if not isinstance(state, np.ndarray):
                raise ValueError("Expected state as a NumPy array, got: {}".format(state))
            if len(state) != input_size:
                raise ValueError("Expected state shape ({},), got {}".format(input_size, state.shape))

            state = torch.tensor(state, device=agent.device, dtype=torch.float32).unsqueeze(0)
            episode_score = 0
            if episode < 1 or episode == num_episodes - 1:
                video_recorder = VideoRecorder(env, path=os.path.join("training", "episode_{}.mp4".format(episode + 1)),
                                               enabled=True)
            while True:
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action.item())
                episode_score += reward

                if terminated is None:
                    next_state_tensor = torch.zeros((1, input_size), device=agent.device, dtype=torch.float32)
                else:
                    next_state_tensor = torch.tensor(next_state, device=agent.device, dtype=torch.float32).unsqueeze(0)

                reward = torch.tensor([reward], device=agent.device, dtype=torch.float32)
                done = terminated or truncated
                agent.memory.push(state, action, next_state_tensor, reward)
                agent.optimize_model()
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * agent.tau + target_net_state_dict[key] * (
                            1 - agent.tau)
                agent.target_net.load_state_dict(target_net_state_dict)
                state = next_state_tensor

                if episode < 1 or episode == num_episodes - 1:
                    video_recorder.capture_frame()

                if done:
                    episode_scores.append(episode_score)
                    episode_result = "Episode {}: Score = {}".format(episode + 1, episode_score)
                    print(episode_result)
                    results_file.write(episode_result + '\n')
                    if episode < 1 or episode == num_episodes - 1:
                        video_recorder.close()
                    break

    plot_learning_curve(episode_scores)


if __name__ == "__main__":
    main()
