import random

class RewardCalculator:
    def __init__(self, train_data, inverse_frequency_too_few, inverse_frequency_too_many):
        self.train_data = train_data
        self.inverse_frequency_too_few = inverse_frequency_too_few
        self.inverse_frequency_too_many = inverse_frequency_too_many

    def get_adjusted_reward(self, state, action, week):
        subset = self.train_data[
            (self.train_data['scale_type'] == state[0]) &
            (self.train_data['weight_bin'] == state[1]) &
            (self.train_data['box_type'] == state[2]) &
            (self.train_data['week'] == week)
        ]
        if subset.empty:
            return -5

        subset = subset.copy()
        subset['sensitivity_diff'] = (subset['sensitivity'] - action).abs()
        closest_row = subset[subset['sensitivity_diff'] == subset['sensitivity_diff'].min()].sample(n=1)
        weight_classification = closest_row['weightclassification'].values[0]

        penalty = {-5: -200, -1: -10, 3: 10}.get(weight_classification, 0)
        frequency_factor = self.inverse_frequency_too_few.get(tuple(state), 1) if weight_classification == -5 else (
            self.inverse_frequency_too_many.get(tuple(state), 1) if weight_classification == -1 else 1
        )
        reward = penalty * (frequency_factor * 0.5)
        return reward
