# 📦 Q-Learning for Supply Chain Sensor Optimization

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository implements an offline, value-based reinforcement learning approach to optimize accelerometer sensitivity settings in supply chain scales. The task involves a trade-off between over-triggering (TOO_MANY) and under-triggering (TOO_FEW) weight measurements. By selecting sensitivity values conditioned on operational context—defined by scale_type, weight_bin, and box_type—the algorithm aims to reduce critical measurement errors, account for hardware-specific behavior, balance state representation, and minimize unnecessary data transmissions, thereby helping to preserve sensor battery life.
---

##  Overview

- **States:** `(scale_type, weight_bin, box_type)`  
- **Actions:** Sensitivity levels `[2, 3, 4, 5, 6, 7, 8]`  
- **Reward:** Weekly, with penalties for critical errors (TOO_FEW) and inverse frequency weighting.  
- **Episodes:** ~1000 (one episode = all weeks in the dataset)  
- **Hyperparameters:**  

| Parameter | Value |
|-----------|-------|
| Alpha     | 0.2   |
| Gamma     | 0.85  |
| Epsilon   | 1.0   |
| Epsilon min | 0.1 |
| Epsilon decay | 0.9985 |

- **Goal:** Learn optimal sensitivity per state to maximize classification accuracy.

---

##  Method

1- For each operational context (state), actions are selected using an epsilon-greedy strategy to balance exploration and exploitation.

2- Since the environment is observational, rewards are computed using an offline counterfactual approximation, where the chosen sensitivity is matched to the closest observed sensitivity in the historical data.

3- Measurement outcomes are mapped to reward values, with severe penalties applied to TOO_FEW events to reflect their higher operational cost. Less frequent state–error combinations are upweighted using inverse-frequency weighting.

4- State–action values are updated using temporal-difference–style value updates under a fixed-state (self-transition) assumption, effectively optimizing action selection within each context.

5- To improve stability and efficiency, the Q-table is pruned after each episode to retain only the top-N actions per state.

---


## Notes on Algorithm

Although Q-learning is traditionally used for sequential decision-making problems with explicit state transitions, sensitivity calibration in this setting is quasi-static and context-dependent, with minimal temporal coupling between decisions. As a result, the problem is more accurately modeled as a contextual bandit or static policy optimization task, where Q-learning serves as a stable and interpretable value-based optimization framework. This formulation also enables straightforward extension to online or adaptive reinforcement learning in future work.


##  Folder Structure

├── data/ # Raw & processed datasets

├── notebooks/ # Jupyter notebook with pipeline

│ └── qlearning_pipeline.ipynb

├── src/ # Modular Python code

│ ├── data_loader.py

│ ├── plot_utils.py

│ ├── qlearning_agent.py

│ └── reward_calculator.py

├── plots/ # plot images

├── requirements.txt # Dependencies

└── README.md


---

##  Installation

git clone https://github.com/irmakoz1/IoT_Sensor_Optimization_with_Reinforcement_Learning.git

cd IoT_Sensor_Optimization_with_Reinforcement_Learning

pip install -r requirements.txt

---
##  Usage

-Raw data is not uploaded to git.

-You can start from the processed data.

- Update DATA_PATH in the notebook or config.

- Open the notebook:

    jupyter notebook notebooks/qlearning_pipeline.ipynb

- Run all cells to train the agent and generate visualizations.

## Results Summary

<img width="1200" height="600" alt="optimal_sensitivity_scale_box_type" src="https://github.com/user-attachments/assets/929ea323-3f9c-4925-b69a-16d46f122ac0" />


<img width="1000" height="600" alt="mse_plot" src="https://github.com/user-attachments/assets/46d6b291-051f-48a8-9fa7-369ec58d3f42" />


<img width="1000" height="600" alt="episode_rewards_with_moving_average" src="https://github.com/user-attachments/assets/7dfce492-8b14-4df0-9452-cbbcdebcb9d6" />


<img width="640" height="480" alt="validation_learning_eval" src="https://github.com/user-attachments/assets/509b27a9-13af-4500-9059-7ff52ccd808c" />




**1-** Sensitivity varies with scale type, weight bin, and box type.

**2-** Lighter bins → higher sensitivity; heavier bins → lower sensitivity.

**3-** Post-training evaluation shows almost all weeks meet criteria.

**4-** The agent learned to balance noise reduction and responsiveness.

## References

Ashiquzzaman, A. et al. (2020) ‘Energy-efficient IOT sensor calibration with Deep Reinforcement
Learning’, IEEE Access, 8, pp. 97045–97055. doi:10.1109/access.2020.2992853. ​

Prauzek, M. and Konecny, J. (2021) ‘Optimizing of Q-learning day/night energy strategy for solar
harvesting environmental wireless sensor networks nodes’, Elektronika ir Elektrotechnika, 27(3), pp. 50–
56. doi:10.5755/j02.eie.28875. ​

Shurrab, M. et al. (2022) ‘IOT sensor selection for target localization: A reinforcement learning based
approach’, Ad Hoc Networks, 134, p. 102927. doi:10.1016/j.adhoc.2022.102927. ​

Wen, Z., O’Neill, D. and Maei, H. (2015) ‘Optimal demand response using device-based reinforcement
learning’, IEEE Transactions on Smart Grid, 6(5), pp. 2312–2324. doi:10.1109/tsg.2015.2396993. 

