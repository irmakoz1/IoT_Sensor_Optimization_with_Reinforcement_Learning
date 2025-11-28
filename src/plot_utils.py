import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlotUtils:

    @staticmethod
    def plot_episode_rewards(episode_rewards, window_size=10):
        smoothed = pd.Series(episode_rewards).rolling(window=window_size).mean()
        plt.figure(figsize=(10,6))
        plt.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
        plt.plot(smoothed, color='red', label=f'Smoothed (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards with Moving Average Smoothing')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_weekly_rewards(weekly_rewards):
        plt.figure(figsize=(12,6))
        plt.plot(weekly_rewards, label='Weekly Average Reward', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Weekly Average Reward')
        plt.title('Weekly Reward Progression')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_q_value_variance(q_value_variance):
        plt.figure(figsize=(12,6))
        plt.plot(q_value_variance, label='Q-value Variance', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Variance in Q-values')
        plt.title('Variance of Q-values Over Episodes')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_epsilon_decay(epsilon_values):
        plt.figure(figsize=(12,6))
        plt.plot(epsilon_values, label='Epsilon Decay', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Episodes')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_optimal_sensitivity(results_df, hue_col='scale_type', kind='bar'):
        plt.figure(figsize=(12,6))
        if kind == 'bar':
            sns.barplot(x='weight_bin', y='optimal_sensitivity', hue=hue_col, data=results_df, palette='Set2')
            plt.title(f"Optimal Sensitivity by {hue_col.capitalize()}")
        elif kind == 'box':
            sns.boxplot(x='weight_bin', y='optimal_sensitivity', hue=hue_col, data=results_df, palette='Set2')
            plt.title(f"Optimal Sensitivity Distribution by {hue_col.capitalize()}")
        plt.xlabel("Weight Bin")
        plt.ylabel("Optimal Sensitivity")
        plt.show()

    @staticmethod
    def plot_boxplot_optimal_sensitivity(results_df):
        plt.figure(figsize=(12,6))
        sns.boxplot(x='weight_bin', y='optimal_sensitivity', hue='scale_type', data=results_df, palette='Set2')
        plt.title("Optimal Sensitivity Boxplot by Scale Type")
        plt.xlabel("Weight Bin")
        plt.ylabel("Optimal Sensitivity")
        plt.show()

    @staticmethod
    def plot_meets_criteria_comparison(df_validation, df_learning):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        sns.countplot(ax=axes[0], x='week', hue='meets_criteria', data=df_validation, palette='Set1')
        axes[0].set_title("Validation Evaluation: Pre-Training")
        axes[0].set_xlabel("Week")
        axes[0].set_ylabel("Number of Entries")
        axes[0].legend(title='Meets Criteria', labels=['No','Yes'])
        PlotUtils.add_counts_to_bars(axes[0])

        sns.countplot(ax=axes[1], x='week', hue='meets_criteria', data=df_learning, palette='Set1')
        axes[1].set_title("Learning Evaluation: Post-Training")
        axes[1].set_xlabel("Week")
        axes[1].set_ylabel("Number of Entries")
        axes[1].legend(title='Meets Criteria', labels=['No','Yes'])
        PlotUtils.add_counts_to_bars(axes[1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sensitivity_distribution(df, weight_bins, hue_col='scale_type'):
        for w_bin in weight_bins:
            plt.figure(figsize=(10,6))
            sns.histplot(data=df[df['weight_bin']==w_bin], x='sensitivity', hue=hue_col, kde=True)
            plt.title(f"Sensitivity Distribution for '{w_bin}' Bin")
            plt.show()

    @staticmethod
    def add_counts_to_bars(ax):
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
