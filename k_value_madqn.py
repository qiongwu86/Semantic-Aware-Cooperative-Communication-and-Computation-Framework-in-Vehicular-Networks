"""
K值选择的多智能体DQN (MADQN) 训练框架

目标: 在SNR ∈ [-10, 20] dB范围内，为所有链路自适应选择k值，最大化语义传输速率
策略: 参数共享的DQN，每个链路（V2V/V2I，来自任何任务车辆）作为一个智能体，共享同一Q网络
状态: SNR, 归一化距离, 归一化任务大小, 链路类型, 语义表对应SNR的20维切片
动作: k ∈ {1,2,...,20}
奖励: 归一化语义传输速率（线性）
"""

import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple, Optional
import math
import pickle

from HighwayEnvironment import HighwayEnvironment


class SumTree:
    """用于优先级采样的Sum Tree数据结构"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的树结构
        self.data = np.zeros(capacity, dtype=object)  # 存储实际数据
        self.write = 0  # 写指针
        self.n_entries = 0  # 当前存储的数据数量
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """检索对应的数据索引"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]
    
    def add(self, p, data):
        """添加数据和对应的优先级"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        """更新指定索引的优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        """根据采样值获取数据"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, buffer_size: int = 100000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.00005):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样权重指数
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # 避免零优先级
        self.max_priority = 1.0  # 最大优先级
    
    def add(self, obs, action, reward, next_obs, done, action_mask, next_action_mask):
        """添加经验，初始优先级设为最大值"""
        experience = (obs, action, reward, next_obs, done, action_mask, next_action_mask)
        self.tree.add(self.max_priority, experience)
    
    def sample(self, batch_size: int):
        """优先级采样"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # 增加beta值（降低增长速度）
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # 在每个段内均匀采样
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # 归一化
        
        # 转换为tensor
        obs_batch = torch.stack([exp[0] for exp in batch])
        action_batch = torch.stack([exp[1] for exp in batch])
        reward_batch = torch.stack([exp[2] for exp in batch])
        next_obs_batch = torch.stack([exp[3] for exp in batch])
        done_batch = torch.stack([exp[4] for exp in batch])
        action_mask_batch = torch.stack([exp[5] for exp in batch])
        next_action_mask_batch = torch.stack([exp[6] for exp in batch])
        
        return (obs_batch, action_batch, reward_batch, next_obs_batch, 
                done_batch, action_mask_batch, next_action_mask_batch, 
                idxs, torch.tensor(is_weights, dtype=torch.float32))
    
    def update_priorities(self, idxs, priorities):
        """更新采样经验的优先级"""
        for idx, priority in zip(idxs, priorities):
            priority = abs(priority) + self.epsilon
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def size(self):
        """返回缓冲区大小"""
        return self.tree.n_entries
    
    def clear(self):
        """清空缓冲区"""
        self.tree = SumTree(self.buffer_size)
        self.max_priority = 1.0


class DQNNetwork(nn.Module):
    """所有链路共享的DQN网络"""
    
    def __init__(self, obs_dim: int = 24, action_dim: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim  # SNR(1) + distance(1) + task_size(1) + link_type(1) + delta_slice(20) = 24
        self.action_dim = action_dim  # k ∈ {1,2,...,20}
        self.hidden_dim = hidden_dim
        
        # Q网络结构
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出每个动作的Q值
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """前向传播"""
        return self.q_network(x)
    
    def get_q_values(self, obs, action_mask=None):
        """获取Q值，考虑动作掩码"""
        q_values = self.forward(obs)
        
        if action_mask is not None:
            # 对无效动作设置非常小的Q值
            q_values = q_values + (action_mask - 1) * 1e9
        
        return q_values


class BasicReplayBuffer:
    """基础经验回放缓冲区（均匀随机采样）"""
    
    def __init__(self, buffer_size: int = 100000):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def add(self, obs, action, reward, next_obs, done, action_mask, next_action_mask):
        """添加一条经验"""
        experience = (obs, action, reward, next_obs, done, action_mask, next_action_mask)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """随机采样一批经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        obs_batch = torch.stack([exp[0] for exp in batch])
        action_batch = torch.stack([exp[1] for exp in batch])
        reward_batch = torch.stack([exp[2] for exp in batch])
        next_obs_batch = torch.stack([exp[3] for exp in batch])
        done_batch = torch.stack([exp[4] for exp in batch])
        action_mask_batch = torch.stack([exp[5] for exp in batch])
        next_action_mask_batch = torch.stack([exp[6] for exp in batch])
        
        # 为了保持接口一致，返回dummy的索引和权重
        idxs = list(range(batch_size))
        is_weights = torch.ones(batch_size, dtype=torch.float32)
        
        return (obs_batch, action_batch, reward_batch, next_obs_batch, 
                done_batch, action_mask_batch, next_action_mask_batch, 
                idxs, is_weights)
    
    def update_priorities(self, idxs, priorities):
        """基础缓冲区不需要更新优先级"""
        pass
    
    def size(self):
        """返回缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class KValueMADQN:
    """K值选择的MADQN训练器 - 所有链路共享同一Q网络"""
    
    def __init__(self, 
                 obs_dim: int = 24,
                 action_dim: int = 20,
                 hidden_dim: int = 256,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 1200000,  
                 epsilon_decay_type: str = "exponential",  # "linear" or "exponential"
                 target_update_freq: int = 1000,
                 batch_size: int = 256,
                 buffer_size: int = 100000,
                 min_buffer_size: int = 5000,  # 增加最小缓冲区大小
                 use_prioritized_replay: bool = True,  # PER开关
                 per_alpha: float = 0.6,
                 per_beta: float = 0.4,
                 per_beta_increment: float = 0.00005,  # 降低beta增长速度
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_type = epsilon_decay_type
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.use_prioritized_replay = use_prioritized_replay
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_network = DQNNetwork(obs_dim, action_dim, hidden_dim).to(device)
        
        # 初始化目标网络参数
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区（根据开关选择类型）
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                alpha=per_alpha,
                beta=per_beta,
                beta_increment=per_beta_increment
            )
            print(f"使用优先级经验回放 (PER) - Alpha: {per_alpha}, Beta: {per_beta}")
        else:
            self.replay_buffer = BasicReplayBuffer(buffer_size)
            print("使用基础经验回放")
        
        # 统计信息
        self.total_steps = 0
        self.training_steps = 0
        self.losses = []  # 存储loss历史
        self.q_values = []  # 存储Q值历史
    
    def get_current_epsilon(self):
        """获取当前epsilon值（改进的epsilon-greedy策略）"""
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        # 计算衰减进度
        decay_ratio = self.total_steps / self.epsilon_decay_steps
        
        if self.epsilon_decay_type == "exponential":
            # 指数衰减：前期快速探索，后期注重模型决策
            # 使用指数函数：epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-3 * decay_ratio)
            # 减小指数参数从5到3，使衰减更缓慢
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-3 * decay_ratio)
        else:
            # 线性衰减（原始方法）
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio
        
        return max(epsilon, self.epsilon_end)  # 确保不低于最小值
    
    def build_observation(self, snr_dB: float, distance: float, task_size: float, 
                         link_type: int, semantic_table: np.ndarray) -> torch.Tensor:
        """构建观测向量（与MAPPO保持一致）"""
        # 归一化SNR到[0,1]: [-10, 20] -> [0, 1]
        snr_norm = np.clip((snr_dB + 10) / 30, 0, 1)
        
        # 归一化距离到[0,1]: [0, 500m] -> [0, 1]
        distance_norm = np.clip(distance / 500, 0, 1)
        
        # 归一化任务大小到[0,1]: [0.1, 1.0] MB -> [0, 1]
        task_size_norm = np.clip((task_size - 0.1) / 0.9, 0, 1)
        
        # 链路类型编码：V2V=0, V2I=1
        link_type_norm = float(link_type)
        
        # 语义表切片：取对应SNR的20维数据
        # 语义表维度：20×30，信噪比是列标记
        snr_idx = int(round(np.clip(snr_dB + 10, 0, 29)))  # SNR范围[-10, 19]映射到[0, 29]
        delta_slice = semantic_table[:, snr_idx]  # 20维（所有k值在该SNR下的语义相似度）
        
        # 组合观测向量
        obs = np.concatenate([
            [snr_norm, distance_norm, task_size_norm, link_type_norm],
            delta_slice
        ])
        
        return torch.tensor(obs, dtype=torch.float32, device=self.device)
    
    def build_action_mask(self, snr_dB: float) -> torch.Tensor:
        """构建动作掩码（与MAPPO保持一致）"""
        mask = torch.ones(20, dtype=torch.float32, device=self.device)
        
        # SNR过低时，限制高k值
        if snr_dB < -5:
            mask[9:] = 0  # 只允许k=1-10
        elif snr_dB < 0:
            mask[14:] = 0  # 只允许k=1-15
        elif snr_dB > 15:
            mask[0] = 0   # 高SNR时不选择k=1
        elif snr_dB > 18:
            mask[19] = 1
        
        return mask
    
    def select_action(self, obs, action_mask, training=True):
        """选择动作（epsilon-greedy策略）"""
        if training and random.random() < self.get_current_epsilon():
            # ✓ 随机探索动作选择正确：
            # - action_idx ∈ [0, 19] 对应 k值 ∈ [1, 20]
            # - 在 _process_link 中：chosen_k = action_idx + 1
            # - 这样k=1对应表格第0行，k=20对应表格第19行，符合要求
            valid_actions = torch.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                action_idx = random.choice(valid_actions.cpu().numpy())
            else:
                action_idx = random.randint(0, 19)
            
            return torch.tensor(action_idx, device=self.device), None
        else:
           with torch.no_grad():
                q_values = self.q_network.get_q_values(obs.unsqueeze(0), action_mask.unsqueeze(0))
                
                # 应用mask
                masked_q_values = q_values.clone()
                mask_bool = action_mask.unsqueeze(0).bool()
                masked_q_values[~mask_bool] = float('-inf')
                
                
                # 重新应用mask确保无效动作仍然被屏蔽
                masked_q_values[~mask_bool] = float('-inf')
                
                # 添加噪声后选择最大Q值对应的动作
                action_idx = masked_q_values.argmax(dim=1).squeeze()
                
            return action_idx, q_values.squeeze()
    
    def compute_reward(self, snr_dB: float, k_value: int, task_size: float, 
                      link_type: int, env_instance) -> Tuple[float, bool, float]:
        """
        计算基于当前链路传输时延的奖励（时延越小奖励越高）
        简化版本：只考虑当前链路的传输时延，根据链路类型选择数据量
        返回: (delay_reward, train_mask, transmission_delay)
        """
        # 判断是否在训练范围内
        train_mask = -10 <= snr_dB <= 20
        
        if not train_mask:
            # SNR范围外，返回0奖励但不用于训练
            return 0.0, False, 0.0
        
        # 固定分配比例（参考test_main_optimized.py的设置）
        lambda_edge = 0.25  # 边缘服务器25%
        lambda_bs = 0.25    # 基站25%
        
        # 根据链路类型确定数据量
        if link_type == 0:  # V2V链路
            link_task_size = task_size * lambda_edge
        else:  # V2I链路 (link_type == 1)
            link_task_size = task_size * lambda_bs
        
        # 计算当前k值的传输时延
        _, current_delay = env_instance.calculate_semantic_rate_and_delay(
            snr_dB, k_value, link_task_size
        )
        
        # 找到最优k值的最小传输时延
        min_delay = float('inf')
        for k in range(1, 21):
            _, delay = env_instance.calculate_semantic_rate_and_delay(
                snr_dB, k, link_task_size
            )
            if delay < min_delay:
                min_delay = delay
        
        # 基于传输时延的奖励设计：时延越接近最优时延奖励越高
        if min_delay > 0 and current_delay > 0:
            # 使用倒数关系：奖励 = min_delay / current_delay
            # 当current_delay = min_delay时，奖励 = 1（最高奖励）
            # 当current_delay > min_delay时，奖励 < 1
            delay_reward = min_delay / current_delay
            delay_reward = np.clip(delay_reward, 0.0, 1.0)  # 限制在[0,1]范围内
        else:
            delay_reward = 0.0
        
        return float(delay_reward), True, float(current_delay)
    
    def _compute_nonlinear_reward(self, linear_norm: float) -> float:
        """
        计算非线性分段奖励函数 (已注释，改用线性归一化速率)
        设计思路：让归一化值越接近1，奖励上升越快，激励智能体探索接近最优的策略
        """
        # 注释掉原有的非线性奖励计算，直接返回线性归一化值
        # x = np.clip(linear_norm, 0.0, 1.0)
        #
        # if x < 0.5:
        #     # 低性能区域：线性增长，斜率0.3
        #     reward = 0.3 * x
        # elif x < 0.8:
        #     # 中等性能区域：二次增长（增加陡峭程度）
        #     # 连续性：在x=0.5时，reward = 0.15
        #     # 设计：reward = 0.15 + a*(x-0.5)^2，增大a值提高区分度
        #     a = 2.5  # 调节参数，使曲线更陡峭
        #     reward = 0.15 + a * (x - 0.5) ** 2
        # elif x < 0.9:
        #     # 高性能区域：指数增长（从0.8到0.9）
        #     # 连续性：在x=0.8时，reward = 0.375
        #     base_reward_08 = 0.375  # 连续点
        #     b = 0.12  # 基础增量
        #     c = 8.0   # 指数系数
        #     reward = base_reward_08 + b * (np.exp(c * (x - 0.8)) - 1)
        # else:
        #     # 最优性能区域：超线性增长（从0.9开始）
        #     # 连续性：在x=0.9时计算base_reward
        #     base_reward = 0.375 + 0.12 * (np.exp(8.0 * (0.9 - 0.8)) - 1)
        #     remaining = 1.0 - base_reward
        #     progress = (x - 0.9) / 0.1
        #     # 使用三次函数实现快速上升
        #     reward = base_reward + remaining * (3 * progress**2 - 2 * progress**3)
        #
        # return float(reward)
        
        # 直接返回线性归一化值作为奖励
        return float(np.clip(linear_norm, 0.0, 1.0))
    
    def update(self):
        """更新Q网络（支持优先级经验回放）"""
        if self.replay_buffer.size() < self.min_buffer_size:
            return None
        
        # 采样批量数据
        batch = self.replay_buffer.sample(self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, action_mask_batch, next_action_mask_batch, idxs, is_weights = batch
        
        # 将重要性采样权重移到设备
        is_weights = is_weights.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network.get_q_values(obs_batch, action_mask_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network.get_q_values(next_obs_batch, next_action_mask_batch)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch.float()))
        
        # 计算TD误差
        td_errors = current_q_values - target_q_values
        
        # 计算加权损失（重要性采样）
        if self.use_prioritized_replay:
            loss = (is_weights * td_errors.pow(2)).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新优先级（如果使用PER）
        if self.use_prioritized_replay:
            priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(idxs, priorities)
        
        # 更新目标网络
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            if self.use_prioritized_replay:
                print(f"步骤 {self.training_steps}: 更新目标网络，当前beta={self.replay_buffer.beta:.3f}")
        
        self.training_steps += 1
        
        # 记录统计信息
        loss_value = loss.item()
        q_value_mean = current_q_values.mean().item()
        
        self.losses.append(loss_value)
        self.q_values.append(q_value_mean)
        
        # debug信息将在episode级别输出，不在这里输出
        
        return {
            'loss': loss.item(),
            'avg_q_value': current_q_values.mean().item(),
            'avg_td_error': abs(td_errors.mean().item()),
            'epsilon': self.get_current_epsilon(),
            'training_steps': self.training_steps,
            'per_beta': self.replay_buffer.beta if self.use_prioritized_replay else 0.0,
            # 添加调试信息到返回字典中
            'current_q_min': current_q_values.min().item(),
            'current_q_max': current_q_values.max().item(),
            'target_q_min': target_q_values.min().item(),
            'target_q_max': target_q_values.max().item(),
            'reward_min': reward_batch.min().item(),
            'reward_max': reward_batch.max().item(),
            'td_error_min': td_errors.min().item(),
            'td_error_max': td_errors.max().item()
        }
    
    def should_update(self):
        """判断是否应该更新网络"""
        return self.replay_buffer.size() >= self.min_buffer_size


class KValueTrainingEnvironmentDQN:
    """K值选择的DQN训练环境包装器"""
    
    def __init__(self, n_task_vehicles: int = 20, n_service_vehicles: int = 5, 
                 enable_lp: bool = True, enable_building_loss: bool = False,
                 use_prioritized_replay: bool = True,
                 epsilon_decay_type: str = "exponential"):
        self.env = HighwayEnvironment(
            n_task_vehicles=n_task_vehicles,
            n_service_vehicles=n_service_vehicles,
            enable_lp=enable_lp
        )
        self.env.enable_building_loss = enable_building_loss
        
        # MADQN训练器
        self.madqn = KValueMADQN(
            use_prioritized_replay=use_prioritized_replay,
            epsilon_decay_type=epsilon_decay_type
        )
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.total_rewards = []
        self.total_performance_ratios = []  # 改为存储性能比值（最小时延/当前时延）
        self.update_count = 0  # 网络更新次数
        
        # 损失统计
        self.loss_history = []
        self.q_value_history = []
        self.epsilon_history = []
        self.td_error_history = []  # TD误差历史
        self.per_beta_history = []  # PER beta历史
        
        # 前一步的状态（用于经验回放）
        self.prev_obs = {}
        self.prev_actions = {}
        
    def train_episode(self, max_steps: int = 100) -> Dict:
        """训练一个回合"""
        episode_reward = 0
        episode_performance_ratios = []
        step_count = 0
        total_links_processed = 0
        
        # 清空前一步状态
        self.prev_obs.clear()
        self.prev_actions.clear()
        
        for step in range(max_steps):
            # 执行一步仿真
            results = self.env.step()
            
            if not results:
                continue
                
            step_count += 1
            
            # 处理每个任务的所有链路（V2V和V2I）
            for result in results:
                valid_links = 0  # 记录有效链路数
                
                # V2V链路智能体
                v2v_obs, v2v_reward, v2v_train_mask, v2v_chosen_k, v2v_transmission_delay = self._process_link(
                    result, 'v2v', result['snr_v2v'], result['k_v2v']
                )
                total_links_processed += 1
                if v2v_train_mask:  # 只有在SNR范围内才有效
                    valid_links += 1
                    episode_performance_ratios.append(v2v_reward)  # 存储性能比值（reward就是min_delay/current_delay）
                
                # V2I链路智能体  
                v2i_obs, v2i_reward, v2i_train_mask, v2i_chosen_k, v2i_transmission_delay = self._process_link(
                    result, 'v2i', result['snr_v2i'], result['k_v2i']
                )
                total_links_processed += 1
                if v2i_train_mask:  # 只有在SNR范围内才有效
                    valid_links += 1
                    episode_performance_ratios.append(v2i_reward)  # 存储性能比值（reward就是min_delay/current_delay）
                
                episode_reward += v2v_reward + v2i_reward
                
                # 增加总步数 (修复: 每个链路只增加1步而不是valid_links步)
                self.madqn.total_steps += 1
                
                # 检查是否需要更新网络（降低更新频率）
                if self.madqn.should_update() and step_count % 20 == 0:  # 每20步检查一次更新
                    update_result = self.madqn.update()
                    if update_result:
                        self.update_count += 1
                        self.loss_history.append(update_result['loss'])
                        self.q_value_history.append(update_result['avg_q_value'])
                        self.epsilon_history.append(update_result['epsilon'])
                        if 'avg_td_error' in update_result:
                            self.td_error_history.append(update_result['avg_td_error'])
                        if 'per_beta' in update_result:
                            self.per_beta_history.append(update_result['per_beta'])
                        
                        # debug信息将在episode级别与统计信息一起输出
                        pass
        
        # 更新统计信息
        if episode_performance_ratios:
            self.total_performance_ratios.extend(episode_performance_ratios)
        
        # 添加回合奖励到历史记录（参考MAPPO）
        self.total_rewards.append(episode_reward)
        
        self.episode_count += 1
        self.step_count += step_count
        
        return {
            'episode': self.episode_count,
            'episode_reward': episode_reward,
            'episode_steps': step_count,
            'total_links': total_links_processed,
            'valid_links': len(episode_performance_ratios),  # 有效训练链路数
            'update_count': self.update_count,
            'avg_performance_ratio': np.mean(episode_performance_ratios) if episode_performance_ratios else 0,
            'current_epsilon': self.madqn.get_current_epsilon(),
            'buffer_size': self.madqn.replay_buffer.size()
        }
    
    def _process_link(self, result: Dict, link_type: str, snr: float, k: int) -> Tuple[torch.Tensor, float, bool, int, float]:
        """处理单个链路"""
        # ✓ SNR过滤正确实现：只处理SNR范围在[-10, 20]之间的链路
        # 这与您的要求一致："保存经验时只保存SNR范围在-10到20之间的"
        train_mask = -10 <= snr <= 20
        if not train_mask:
            # SNR范围外，不存储经验，直接返回
            return None, 0.0, False, k, 0.0
        
        # 计算距离
        task_vehicle_id = result['task_vehicle_id']
        service_vehicle_id = result['service_vehicle_id']
        
        if link_type == 'v2v':
            # V2V: 任务车辆到服务车辆的距离
            task_pos = self.env.vehicles[task_vehicle_id].position
            service_pos = self.env.vehicles[service_vehicle_id].position
            distance = math.hypot(task_pos[0] - service_pos[0], task_pos[1] - service_pos[1])
            link_type_code = 0
        else:  # v2i
            # V2I: 任务车辆到基站的距离
            task_pos = self.env.vehicles[task_vehicle_id].position
            bs_pos = [self.env.width/2, self.env.height/2]
            distance = math.hypot(task_pos[0] - bs_pos[0], task_pos[1] - bs_pos[1])
            link_type_code = 1
        
        # 构建观测
        task_size = 0.4  # 使用平均任务大小（与MAPPO保持一致）
        obs = self.madqn.build_observation(
            snr, distance, task_size, 
            link_type_code, self.env.semantic_table
        )
        
        # 构建动作掩码
        action_mask = self.madqn.build_action_mask(snr)
        
        # 生成唯一的链路ID
        link_id = f"{task_vehicle_id}_{service_vehicle_id}_{link_type}"
        
        # 处理前一步的经验（如果存在）
        if link_id in self.prev_obs:
            prev_obs = self.prev_obs[link_id]
            prev_action = self.prev_actions[link_id]
            
            # 计算前一步的奖励（使用前一步智能体选择的k值）
            prev_chosen_k = prev_action.item() + 1  # 转换为1-20的k值
            reward, _, transmission_delay = self.madqn.compute_reward(
                snr, prev_chosen_k, task_size, link_type_code, self.env
            )
            
            # 只有SNR在训练范围内才添加到经验回放缓冲区
            prev_snr = prev_obs.get('snr', 0)
            if -10 <= prev_snr <= 20:  # 确保SNR在训练范围内
                # 添加到经验回放缓冲区
                # 对于车辆通信任务，每个任务完成后可以认为done=True
                # 这样可以避免Q值过度传播
                self.madqn.replay_buffer.add(
                    prev_obs['obs'], prev_obs['action'], 
                    torch.tensor(reward, device=self.madqn.device),
                    obs, torch.tensor(True, device=self.madqn.device),  # done=True（任务完成）
                    prev_obs['action_mask'], action_mask
                )
        
        # 选择动作
        action_idx, q_values = self.madqn.select_action(obs, action_mask, training=True)
        chosen_k = action_idx.item() + 1  # 转换为1-20的k值
        
        # 计算奖励和传输时延
        reward, _, transmission_delay = self.madqn.compute_reward(
            snr, chosen_k, task_size, link_type_code, self.env
        )
        
        # 存储当前状态以备下一步使用
        self.prev_obs[link_id] = {
            'obs': obs,
            'action': action_idx,
            'action_mask': action_mask,
            'snr': snr  # 存储SNR用于过滤
        }
        self.prev_actions[link_id] = action_idx
        
        return obs, reward, True, chosen_k, transmission_delay
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线（增强版，包含PER相关指标）"""
        if not self.loss_history:
            print("No training data to plot.")
            return
        
        # 根据是否使用PER决定子图布局
        if self.madqn.use_prioritized_replay and self.td_error_history:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 2x3布局，增加PER相关图
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2布局
        
        # 1. Loss曲线
        axes[0, 0].plot(self.loss_history, alpha=0.7, color='red', linewidth=1)
        # 添加平滑曲线
        if len(self.loss_history) > 50:
            window_size = min(50, len(self.loss_history) // 10)
            smoothed_loss = []
            for i in range(window_size, len(self.loss_history)):
                smoothed_loss.append(np.mean(self.loss_history[i-window_size:i]))
            axes[0, 0].plot(range(window_size, len(self.loss_history)), smoothed_loss, 
                           color='darkred', linewidth=2, label='Smoothed')
        loss_title = 'DQN Loss (PER)' if self.madqn.use_prioritized_replay else 'DQN Loss'
        axes[0, 0].set_title(loss_title, fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Q值曲线
        axes[0, 1].plot(self.q_value_history, alpha=0.7, color='blue', linewidth=1)
        # 添加平滑曲线
        if len(self.q_value_history) > 50:
            window_size = min(50, len(self.q_value_history) // 10)
            smoothed_q = []
            for i in range(window_size, len(self.q_value_history)):
                smoothed_q.append(np.mean(self.q_value_history[i-window_size:i]))
            axes[0, 1].plot(range(window_size, len(self.q_value_history)), smoothed_q,
                           color='darkblue', linewidth=2, label='Smoothed')
        axes[0, 1].set_title('Average Q-Value', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Q-Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Epsilon曲线（显示衰减类型）
        epsilon_title = f'Epsilon ({self.madqn.epsilon_decay_type.title()} Decay)'
        axes[1, 0].plot(self.epsilon_history, color='green', linewidth=2)
        axes[1, 0].set_title(epsilon_title, fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 语义传输速率
        if self.total_semantic_rates:
            # 计算移动平均
            window_size = min(100, len(self.total_semantic_rates) // 10)
            if window_size > 1:
                moving_avg = []
                for i in range(window_size, len(self.total_semantic_rates)):
                    moving_avg.append(np.mean(self.total_semantic_rates[i-window_size:i]))
                axes[1, 1].plot(range(window_size, len(self.total_semantic_rates)), moving_avg,
                               color='purple', linewidth=2)
            else:
                axes[1, 1].plot(self.total_semantic_rates, alpha=0.5, color='purple')
        
        axes[1, 1].set_title('Semantic Transmission Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Experience Steps')
        axes[1, 1].set_ylabel('Normalized Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5&6. PER相关图（如果启用PER）
        if self.madqn.use_prioritized_replay and len(axes.shape) == 2 and axes.shape[1] == 3:
            # 5. TD误差曲线
            if self.td_error_history:
                axes[0, 2].plot(self.td_error_history, alpha=0.7, color='orange', linewidth=1)
                if len(self.td_error_history) > 50:
                    window_size = min(50, len(self.td_error_history) // 10)
                    smoothed_td = []
                    for i in range(window_size, len(self.td_error_history)):
                        smoothed_td.append(np.mean(self.td_error_history[i-window_size:i]))
                    axes[0, 2].plot(range(window_size, len(self.td_error_history)), smoothed_td,
                                   color='darkorange', linewidth=2, label='Smoothed')
                axes[0, 2].set_title('TD Error (PER Priority)', fontsize=14, fontweight='bold')
                axes[0, 2].set_xlabel('Training Steps')
                axes[0, 2].set_ylabel('|TD Error|')
                axes[0, 2].grid(True, alpha=0.3)
                axes[0, 2].legend()
            
            # 6. PER Beta曲线
            if self.per_beta_history:
                axes[1, 2].plot(self.per_beta_history, color='brown', linewidth=2)
                axes[1, 2].set_title('PER Beta (Importance Sampling)', fontsize=14, fontweight='bold')
                axes[1, 2].set_xlabel('Training Steps')
                axes[1, 2].set_ylabel('Beta')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


def main():
    """主训练循环"""
    # 创建训练环境
    n_task_vehicles = 20
    use_per = True  # PER开关
    epsilon_decay = "exponential"  # epsilon衰减策略
    
    train_env = KValueTrainingEnvironmentDQN(
        n_task_vehicles=n_task_vehicles,
        n_service_vehicles=5,
        enable_lp=True,
        enable_building_loss=False,
        use_prioritized_replay=use_per,
        epsilon_decay_type=epsilon_decay
    )
    
    # 输出初始epsilon信息
    print(f"\n初始Epsilon: {train_env.madqn.get_current_epsilon():.4f}")
    print(f"Epsilon衰减步数: {train_env.madqn.epsilon_decay_steps:,}")
    print(f"Epsilon衰减类型: {train_env.madqn.epsilon_decay_type}")
    
    # 训练参数
    total_episodes = 1000
    max_steps_per_episode = 100
    
    print("开始K值选择MADQN训练...")
    print(f"训练回合数: {total_episodes}")
    print(f"每回合最大步数: {max_steps_per_episode}")
    print(f"任务车辆数: {n_task_vehicles}")
    print(f"服务车辆数: 5")
    print(f"优先级经验回放: {'✓ 启用' if use_per else '✗ 禁用'}")
    print(f"探索策略: {epsilon_decay.title()} 衰减")
    print(f"目标网络更新频率: {train_env.madqn.target_update_freq}")
    print(f"批量大小: {train_env.madqn.batch_size}")
    print("=" * 60)
    
    # 训练循环
    for episode in range(total_episodes):
        episode_info = train_env.train_episode(max_steps_per_episode)
        
        # 每10回合打印一次统计信息（与MAPPO保持一致）
        if episode % 10 == 0:
            # 计算平均奖励和平均速率（参考MAPPO）
            avg_reward = np.mean(train_env.total_rewards[-10:]) if len(train_env.total_rewards) >= 10 else (np.mean(train_env.total_rewards) if train_env.total_rewards else 0)
            # 删除这行，不再需要avg_rate
            
            # 基础统计信息（参考MAPPO格式）
            avg_performance = np.mean(train_env.total_performance_ratios[-100:]) if len(train_env.total_performance_ratios) >= 100 else 0
            
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_info['episode_reward']:6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Perf: {avg_performance:.4f} | "
                  f"Steps: {episode_info['episode_steps']:3d} | "
                  f"Links: {episode_info['valid_links']:3d} | "
                  f"TotalSteps: {train_env.madqn.total_steps:5d} | "
                  f"Updates: {episode_info['update_count']:3d}")
            
            # 损失信息（类似MAPPO的policy loss等）
            if train_env.loss_history:
                recent_loss = np.mean(train_env.loss_history[-10:]) if len(train_env.loss_history) >= 10 else train_env.loss_history[-1]
                recent_q_val = np.mean(train_env.q_value_history[-10:]) if len(train_env.q_value_history) >= 10 else (train_env.q_value_history[-1] if train_env.q_value_history else 0)
                
                # 检查损失值是否异常小
                if recent_loss < 1e-6:
                    loss_str = f"{recent_loss:.2e}"  # 科学计数法显示很小的值
                else:
                    loss_str = f"{recent_loss:8.4f}"
                
                loss_info = (f"           | "
                           f"DQN Loss: {loss_str:>10s} | "
                           f"Avg Q-Val: {recent_q_val:8.4f} | "
                           f"Epsilon: {episode_info['current_epsilon']:.6f} | "
                           f"Buffer: {episode_info['buffer_size']:5d}")
                
                # PER相关信息
                if use_per and train_env.per_beta_history:
                    current_beta = train_env.per_beta_history[-1] if train_env.per_beta_history else 0.4
                    avg_td_error = np.mean(train_env.td_error_history[-10:]) if len(train_env.td_error_history) >= 10 else 0
                    loss_info += f" | Beta: {current_beta:.3f} | TD-Err: {avg_td_error:.4f}"
                
                print(loss_info)
        
        # 保存模型（参考MAPPO的100回合保存）
        if episode % 100 == 0 and episode > 0:
            model_suffix = f"_per_{epsilon_decay}" if use_per else f"_basic_{epsilon_decay}"
            save_path = f'k_value_madqn_episode_{episode}_{n_task_vehicles}vehicles{model_suffix}.pth'
            torch.save({
                'q_network': train_env.madqn.q_network.state_dict(),
                'target_network': train_env.madqn.target_network.state_dict(),
                'optimizer': train_env.madqn.optimizer.state_dict(),
                'episode': episode,
                'total_steps': train_env.madqn.total_steps,
                'training_steps': train_env.madqn.training_steps,
                'use_per': use_per,
                'epsilon_decay_type': epsilon_decay
            }, save_path)
            print(f"模型已保存: {save_path}")
        
        # 删除中间绘制，只在最后绘制一次
        # 原来每200回合绘制一次，现在删除
    
    # 保存最终模型
    model_suffix = f"_per_{epsilon_decay}" if use_per else f"_basic_{epsilon_decay}"
    torch.save({
        'q_network': train_env.madqn.q_network.state_dict(),
        'target_network': train_env.madqn.target_network.state_dict(),
        'optimizer': train_env.madqn.optimizer.state_dict(),
        'episode_count': train_env.episode_count,
        'total_steps': train_env.madqn.total_steps,
        'training_steps': train_env.madqn.training_steps,
        'use_per': use_per,
        'epsilon_decay_type': epsilon_decay
    }, f"k_value_madqn_final_{n_task_vehicles}vehicles{model_suffix}.pth")
    
    print("训练完成!")
    
    # 添加训练总结（类似MAPPO）
    print("\n" + "=" * 60)
    print("MADQN训练总结")
    print("=" * 60)
    print(f"总回合数: {train_env.episode_count}")
    print(f"总经验步数: {train_env.madqn.total_steps}")
    print(f"网络更新次数: {train_env.update_count}")
    print(f"最终Epsilon: {train_env.madqn.get_current_epsilon():.4f}")
    if train_env.total_performance_ratios:
        print(f"最终平均性能比值: {np.mean(train_env.total_performance_ratios[-100:]):.4f}")
    if train_env.total_rewards:
        print(f"最终平均奖励: {np.mean(train_env.total_rewards[-10:]):.4f}")
    
    # PER相关统计
    if use_per:
        print(f"最终PER Beta: {train_env.per_beta_history[-1]:.3f}" if train_env.per_beta_history else "PER Beta: N/A")
        if train_env.td_error_history:
            print(f"最终平均TD误差: {np.mean(train_env.td_error_history[-10:]):.4f}")
    
    print(f"最终模型: k_value_madqn_final_{n_task_vehicles}vehicles{model_suffix}.pth")
    print("=" * 60)
    
    # 保存训练数据到文件
    training_data = {
        'loss_history': train_env.loss_history,
        'q_value_history': train_env.q_value_history,
        'epsilon_history': train_env.epsilon_history,
        'td_error_history': train_env.td_error_history,
        'per_beta_history': train_env.per_beta_history if use_per else [],
        'total_rewards': train_env.total_rewards,
        'total_performance_ratios': train_env.total_performance_ratios,
        'training_config': {
            'n_task_vehicles': n_task_vehicles,
            'total_episodes': total_episodes,
            'use_per': use_per,
            'epsilon_decay_type': epsilon_decay,
            'min_buffer_size': train_env.madqn.min_buffer_size,
            'batch_size': train_env.madqn.batch_size,
            'lr': train_env.madqn.optimizer.param_groups[0]['lr'],
            'gamma': train_env.madqn.gamma
        }
    }
    
    data_filename = f"madqn_training_data_{n_task_vehicles}vehicles{model_suffix}.pkl"
    with open(data_filename, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"\n训练数据已保存: {data_filename}")
    
    # 绘制最终训练曲线
    final_plot_name = f"madqn_final_training_curves_{n_task_vehicles}vehicles{model_suffix}.png"
    train_env.plot_training_curves(final_plot_name)


if __name__ == "__main__":
    main()


