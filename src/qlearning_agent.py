import numpy as np
import random
import heapq

class QLearningAgent:
    def __init__(self, fixed_states, actions, reward_calculator, alpha=0.2, gamma=0.85, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9985, top_n=5):
        self.fixed_states = fixed_states
        self.actions = actions
        self.reward_calculator = reward_calculator
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.top_n = top_n
        self.Q_table = {tuple(state): {action: 0 for action in actions} for state in fixed_states}
        self.episode_rewards = []

    def calculate_q_variance(self):
        all_q = [q for state in self.Q_table.values() for q in state.values()]
        return np.var(all_q)

    def prune_q_table(self):
        for state in self.Q_table:
            actions_q = self.Q_table[state]
            top_actions = heapq.nlargest(self.top_n, actions_q, key=actions_q.get)
            for action in list(actions_q.keys()):
                if action not in top_actions:
                    self.Q_table[state][action] = 0

    def train(self, weeks, num_episodes):
        for episode in range(num_episodes):
            total_reward = 0
            week_rewards = []
            for week in weeks:
                for state in self.fixed_states:
                    action = random.choice(self.actions) if random.uniform(0, 1) < self.epsilon else max(self.Q_table[state], key=self.Q_table[state].get)
                    reward = self.reward_calculator.get_adjusted_reward(state, action, week)
                    total_reward += reward
                    week_rewards.append(reward)

                    best_next_action = max(self.Q_table[state], key=self.Q_table[state].get)
                    old_q = self.Q_table[state][action]
                    target = reward + self.gamma * self.Q_table[state][best_next_action]
                    td_error = target - old_q
                    self.Q_table[state][action] = old_q + self.alpha * td_error

            self.prune_q_table()
            self.episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
