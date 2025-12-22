"""
K值选择的多智能体PPO (MAPPO) 训练框架

目标: 在SNR ∈ [-10, 20] dB范围内，为所有链路自适应选择k值，最大化语义传输速率
策略: 参数共享的MAPPO，每个链路（V2V/V2I，来自任何任务车辆）作为一个智能体，共享同一策略网络
状态: SNR, 归一化距离, 归一化任务大小, 链路类型, 语义表对应SNR的20维切片
动作: k ∈ {1,2,...,20}
奖励: 归一化语义传输速率（线性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple, Optional
import math
import pickle

from HighwayEnvironment import HighwayEnvironment


class SharedActorCritic(nn.Module):
    """所有链路共享的Actor-Critic网络"""
    
    def __init__(self, obs_dim: int = 24, action_dim: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim  # SNR(1) + distance(1) + task_size(1) + link_type(1) + delta_slice(20) = 24
        self.action_dim = action_dim  # k ∈ {1,2,...,20}
        self.hidden_dim = hidden_dim
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头：输出动作概率分布
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic头：输出状态价值
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        obs: (batch_size, obs_dim)
        action_mask: (batch_size, action_dim) - 1表示可选，0表示屏蔽
        """
        batch_size = obs.shape[0]
        
        # 共享特征提取
        shared_features = self.shared_layers(obs)
        
        # Actor输出：动作logits
        action_logits = self.actor_head(shared_features)
        
        # 应用动作屏蔽
        if action_mask is not None:
            action_logits = action_logits + (action_mask - 1) * 1e9  # 屏蔽的动作设为极小值
        
        # 动作概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic输出：状态价值
        state_value = self.critic_head(shared_features).squeeze(-1)
        
        return action_probs, state_value
    
    def get_action_and_value(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, 
                           action: Optional[torch.Tensor] = None):
        """获取动作和价值，用于训练"""
        action_probs, state_value = self.forward(obs, action_mask)
        
        # 创建分类分布
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            # 采样动作
            action = dist.sample()
        
        # 计算动作的对数概率和熵
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_logprob, state_value, entropy


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.action_logprobs = []
        self.state_values = []
        self.dones = []
        self.action_masks = []
        self.train_masks = []  # 用于屏蔽SNR范围外的样本
        
    def clear(self):
        """清空缓冲区"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.action_logprobs.clear()
        self.state_values.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.train_masks.clear()
        
    def add(self, obs, action, reward, action_logprob, state_value, done, action_mask, train_mask):
        """添加一步经验"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_logprobs.append(action_logprob)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.train_masks.append(train_mask)
        
    def get(self):
        """获取所有经验"""
        return (
            torch.stack(self.observations),
            torch.stack(self.actions),
            torch.stack(self.rewards),
            torch.stack(self.action_logprobs),
            torch.stack(self.state_values),
            torch.stack(self.dones),
            torch.stack(self.action_masks),
            torch.stack(self.train_masks)
        )


class KValueMAPPO:
    """K值选择的MAPPO训练器 - 所有链路共享同一策略网络"""
    
    def __init__(self, 
                 obs_dim: int = 24,
                 action_dim: int = 20,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.2,
                 ent_coef_init: float = 0.1,  # 初始熵系数，增大促进探索
                 ent_coef_final: float = 0.01,  # 最终熵系数
                 ent_decay_steps: int = 100000,  # 熵衰减步数
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 update_frequency: int = 2048,  # 每2048个经验更新一次
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_init = ent_coef_init
        self.ent_coef_final = ent_coef_final
        self.ent_decay_steps = ent_decay_steps
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency
        
        # 网络
        self.policy = SharedActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_experiences = 0  # 累计经验数量
        self.training_steps = 0  # 训练步数，用于熵衰减
        
    def build_observation(self, snr_dB: float, distance: float, task_size: float, 
                         link_type: int, semantic_table: np.ndarray) -> torch.Tensor:
        """构建观测向量"""
        # 归一化SNR到[0,1]: [-10, 20] -> [0, 1]
        snr_norm = np.clip((snr_dB + 10) / 30, 0, 1)
        
        # 归一化距离到[0,1]: 假设最大距离为400m (道路长度)
        distance_norm = np.clip(distance / 400.0, 0, 1)
        
        # 归一化任务大小到[0,1]: 假设范围[0.3, 0.5] Mbit
        task_size_norm = np.clip((task_size - 0.3) / (0.5 - 0.3), 0, 1)
        
        # 链路类型: V2V=0, V2I=1
        link_type_norm = float(link_type)
        
        # 从语义表获取当前SNR对应的20维切片
        snr_idx = int(round(np.clip(snr_dB + 10, 0, 30)))
        snr_idx = np.clip(snr_idx, 0, semantic_table.shape[1] - 1)  # semantic_table: (20, 31)
        delta_slice = semantic_table[:, snr_idx]  # 20维向量
        
        # 组合观测向量
        obs = np.concatenate([
            [snr_norm, distance_norm, task_size_norm, link_type_norm],
            delta_slice
        ])
        
        return torch.FloatTensor(obs).to(self.device)
    
    def build_action_mask(self, snr_dB: float) -> torch.Tensor:
        """构建动作屏蔽掩码"""
        mask = torch.ones(20, dtype=torch.float32, device=self.device)
        
        # SNR范围外的情况不参与学习，但仍需要掩码
        if snr_dB > 20:
            # 只允许k=1 (索引0)
            mask[:] = 0
            mask[0] = 1
        elif snr_dB < -10:
            # 只允许k=20 (索引19)
            mask[:] = 0
            mask[19] = 1
        # else: 在[-10, 20]范围内，所有动作都可选
        
        return mask
    
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
        
        分段函数设计：
        - [0, 0.5): 线性增长，斜率较小 (惩罚差策略)
        - [0.5, 0.8): 二次增长，适中激励 (鼓励改进)
        - [0.8, 0.9): 指数增长，强烈激励 (奖励优秀策略)
        - [0.9, 1.0]: 超线性增长，最大激励 (追求最优)
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
        #     a = 2.5  # 调节参数，从1.0增加到2.5，使曲线更陡峭
        #     reward = 0.15 + a * (x - 0.5) ** 2
        # elif x < 0.9:
        #     # 高性能区域：指数增长（从0.8到0.9）
        #     # 连续性：在x=0.8时，reward = 0.375
        #     base_reward_08 = 0.375  # 更新的连续点
        #     b = 0.12  # 基础增量，适当调整
        #     c = 8.0   # 指数系数，适当调整
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
    
    def select_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, float, float]:
        """选择动作（训练时使用，带随机性）"""
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0)  # (1, obs_dim)
            action_mask_batch = action_mask.unsqueeze(0)  # (1, action_dim)
            
            action, action_logprob, state_value, _ = self.policy.get_action_and_value(
                obs_batch, action_mask_batch
            )
            
            return action.item(), action_logprob.item(), state_value.item()
    
    def select_best_action(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        """选择最佳动作（测试时使用）"""
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0)
            action_mask_batch = action_mask.unsqueeze(0)
            
            # 获取动作概率分布
            logits, _ = self.policy(obs_batch, action_mask_batch)
            
            # 应用mask
            masked_logits = logits.clone()
            mask_bool = action_mask_batch.bool()
            masked_logits[~mask_bool] = float('-inf')         
            
            # 重新应用mask确保无效动作仍然被屏蔽
            masked_logits[~mask_bool] = float('-inf')
            
            probs = torch.softmax(masked_logits, dim=-1)
        
            # 使用多项分布采样一个动作（batch_size=1）
            action_index = torch.multinomial(probs, num_samples=1).item()
        
            # 转换回 [1, 20] 范围
            return action_index + 1
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                   train_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势估计"""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t].float()
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - dones[t + 1].float()
                nextvalues = values[t + 1]
            
            # 只对训练样本计算GAE
            if train_masks[t]:
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            else:
                advantages[t] = 0
                lastgaelam = 0
        
        returns = advantages + values
        return advantages, returns
    
    def get_current_ent_coef(self) -> float:
        """获取当前的熵系数（随训练步数衰减）"""
        progress = min(self.training_steps / self.ent_decay_steps, 1.0)
        return self.ent_coef_init * (1 - progress) + self.ent_coef_final * progress
    
    def should_update(self) -> bool:
        """判断是否应该更新策略"""
        return len(self.buffer.observations) >= self.update_frequency
    
    def update(self, batch_size: int = 64, update_epochs: int = 10):
        """更新策略"""
        if len(self.buffer.observations) == 0:
            return {}
        
        # 获取经验
        obs, actions, rewards, old_logprobs, values, dones, action_masks, train_masks = self.buffer.get()
        
        # 计算GAE
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, values, dones, train_masks)
            
            # 标准化优势（只对训练样本）
            train_advantages = advantages[train_masks.bool()]
            if len(train_advantages) > 0 and train_advantages.std() > 1e-8:
                advantages[train_masks.bool()] = (train_advantages - train_advantages.mean()) / (train_advantages.std() + 1e-8)
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(update_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                batch_train_masks = train_masks[batch_indices]
                
                # 前向传播（所有样本都是可训练的）
                _, new_logprobs, new_values, entropy = self.policy.get_action_and_value(
                    batch_obs, batch_action_masks, batch_actions
                )
                
                # PPO损失
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失（使用动态熵系数）
                current_ent_coef = self.get_current_ent_coef()
                loss = policy_loss + self.vf_coef * value_loss + current_ent_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # 更新训练步数
        self.training_steps += len(obs)
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / update_epochs,
            'value_loss': total_value_loss / update_epochs,
            'entropy_loss': total_entropy_loss / update_epochs,
            'ent_coef': self.get_current_ent_coef(),
            'training_steps': self.training_steps
        }


class KValueTrainingEnvironment:
    """K值选择的训练环境包装器"""
    
    def __init__(self, n_task_vehicles: int = 20, n_service_vehicles: int = 5, 
                 enable_lp: bool = True, enable_building_loss: bool = False):
        self.env = HighwayEnvironment(
            n_task_vehicles=n_task_vehicles,
            n_service_vehicles=n_service_vehicles,
            enable_lp=enable_lp
        )
        self.env.enable_building_loss = enable_building_loss
        
        # MAPPO训练器
        self.mappo = KValueMAPPO()
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.total_rewards = []
        self.total_performance_ratios = []  # 改为存储性能比值（最小时延/当前时延）
        self.update_count = 0  # 策略更新次数
        
        # 训练历史记录（参照MADQN格式）
        self.loss_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_loss_history = []
        self.ent_coef_history = []
        
    def train_episode(self, max_steps: int = 100) -> Dict:
        """训练一个回合"""
        episode_reward = 0
        episode_performance_ratios = []
        step_count = 0
        total_links_processed = 0
        
        for step in range(max_steps):
            # 执行一步仿真
            results = self.env.step()
            
            if not results:
                continue
                
            step_count += 1
            
            # 处理每个任务的所有链路（V2V和V2I）
            # 每个链路都是一个独立的智能体，但共享同一策略网络
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
                
                # 更新经验计数（只计算有效经验）
                self.mappo.total_experiences += valid_links
                
                # 检查是否需要更新策略
                if self.mappo.should_update():
                    buffer_size = len(self.mappo.buffer.observations)  # 在清空前记录
                    losses = self.mappo.update()
                    self.update_count += 1
                    
                    # 记录损失历史（参照MADQN格式）
                    if 'policy_loss' in losses:
                        self.policy_loss_history.append(losses['policy_loss'])
                        self.value_loss_history.append(losses['value_loss'])
                        self.entropy_loss_history.append(losses['entropy_loss'])
                        self.ent_coef_history.append(losses['ent_coef'])
                        
                        # 计算总损失（与MADQN的loss_history对应）
                        total_loss = losses['policy_loss'] + losses['value_loss'] + losses['entropy_loss']
                        self.loss_history.append(total_loss)
                    
                    print(f"  策略更新 #{self.update_count}, 使用经验: {buffer_size}")
                    # 重置经验计数
                    self.mappo.total_experiences = 0
        
        # Episode结束时的最终更新（如果还有剩余经验）
        if len(self.mappo.buffer.observations) > 0:
            remaining_size = len(self.mappo.buffer.observations)
            losses = self.mappo.update()
            self.update_count += 1
            
            # 记录损失历史（参照MADQN格式）
            if 'policy_loss' in losses:
                self.policy_loss_history.append(losses['policy_loss'])
                self.value_loss_history.append(losses['value_loss'])
                self.entropy_loss_history.append(losses['entropy_loss'])
                self.ent_coef_history.append(losses['ent_coef'])
                
                # 计算总损失（与MADQN的loss_history对应）
                total_loss = losses['policy_loss'] + losses['value_loss'] + losses['entropy_loss']
                self.loss_history.append(total_loss)
            
            print(f"  Episode结束更新 #{self.update_count}, 剩余经验: {remaining_size}")
        else:
            losses = {}
        
        self.episode_count += 1
        self.total_rewards.append(episode_reward)
        if episode_performance_ratios:
            self.total_performance_ratios.extend(episode_performance_ratios)
        
        return {
            'episode': self.episode_count,
            'episode_reward': episode_reward,
            'episode_steps': step_count,
            'total_links': total_links_processed,
            'valid_links': len(episode_performance_ratios),  # 有效训练链路数
            'update_count': self.update_count,
            'avg_performance_ratio': np.mean(episode_performance_ratios) if episode_performance_ratios else 0,
            **losses
        }
    
    def _process_link(self, result: Dict, link_type: str, snr: float, k: int) -> Tuple[torch.Tensor, float, bool, int, float]:
        """处理单个链路"""
        # 首先检查SNR是否在训练范围内，不在范围内直接跳过
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
            link_type_id = 0
        else:  # v2i
            # V2I: 任务车辆到基站的距离
            task_pos = self.env.vehicles[task_vehicle_id].position
            bs_pos = [self.env.width/2, self.env.height/2]
            distance = math.hypot(task_pos[0] - bs_pos[0], task_pos[1] - bs_pos[1])
            link_type_id = 1
        
        # 构建观测
        task_size = 0.4  # 使用平均任务大小
        obs = self.mappo.build_observation(
            snr, distance, task_size, link_type_id, self.env.semantic_table
        )
        
        # 构建动作掩码（SNR在范围内，所有动作都可选）
        action_mask = torch.ones(20, dtype=torch.float32, device=self.mappo.device)
        
        # 让智能体选择k值（而不是使用环境选择的k）
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0)
            action_mask_batch = action_mask.unsqueeze(0)
            
            # 智能体自主选择动作
            action_tensor, action_logprob_tensor, state_value_tensor, _ = self.mappo.policy.get_action_and_value(
                obs_batch, action_mask_batch, None  # None表示让网络自己采样
            )
            
            action = action_tensor.item()  # 0-19索引
            chosen_k = action + 1  # 转换为1-20的k值
            action_logprob = action_logprob_tensor.item()
            state_value = state_value_tensor.item()
        
        # 基于智能体选择的k值计算奖励
        reward, _, transmission_delay = self.mappo.compute_reward(
            snr, chosen_k, task_size, link_type_id, self.env
        )
        
        # reward已经是min_delay/current_delay的比值，直接使用作为性能指标
        
        # 添加到缓冲区（只存储有效的训练样本）
        self.mappo.buffer.add(
            obs, torch.tensor(action, device=self.mappo.device),
            torch.tensor(reward, device=self.mappo.device),
            torch.tensor(action_logprob, device=self.mappo.device),
            torch.tensor(state_value, device=self.mappo.device),
            torch.tensor(False, device=self.mappo.device),  # done
            action_mask,
            torch.tensor(True, device=self.mappo.device)  # 所有存储的样本都是可训练的
        )
        
        return obs, reward, True, chosen_k, transmission_delay


def main():
    """主训练循环"""
    # 训练参数
    n_task_vehicles = 20
    n_service_vehicles = 5
    num_episodes = 1000
    max_steps_per_episode = 100
    
    # 创建训练环境
    train_env = KValueTrainingEnvironment(
        n_task_vehicles=n_task_vehicles,
        n_service_vehicles=n_service_vehicles,
        enable_lp=True,
        enable_building_loss=False
    )
    
    print("开始K值选择MAPPO训练...")
    print(f"训练回合数: {num_episodes}")
    print(f"每回合最大步数: {max_steps_per_episode}")
    print("=" * 60)
    
    # 训练循环
    for episode in range(num_episodes):
        stats = train_env.train_episode(max_steps_per_episode)
        
        # 打印统计信息
        if episode % 10 == 0:
            avg_reward = np.mean(train_env.total_rewards[-10:]) if len(train_env.total_rewards) >= 10 else 0
            avg_performance = np.mean(train_env.total_performance_ratios[-100:]) if len(train_env.total_performance_ratios) >= 100 else 0
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {stats['episode_reward']:6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Perf: {avg_performance:.4f} | "
                  f"Steps: {stats['episode_steps']:3d} | "
                  f"Links: {stats['total_links']:3d} | "
                  f"Valid: {stats['valid_links']:3d} | "
                  f"Updates: {stats['update_count']:3d}")
            
            if 'policy_loss' in stats:
                print(f"           | "
                      f"Policy Loss: {stats['policy_loss']:8.4f} | "
                      f"Value Loss: {stats['value_loss']:8.4f} | "
                      f"Entropy Loss: {stats['entropy_loss']:8.4f} | "
                      f"Ent Coef: {stats.get('ent_coef', 0.01):.4f}")
        
        # 保存模型
        if episode % 100 == 0 and episode > 0:
            torch.save(train_env.mappo.policy.state_dict(), f'k_value_mappo_episode_{episode}.pth')
            print(f"模型已保存: k_value_mappo_episode_{episode}.pth")
    
    print("训练完成!")
    
    # 保存最终模型
    torch.save(train_env.mappo.policy.state_dict(), 'k_value_mappo_final.pth')
    print("最终模型已保存: k_value_mappo_final.pth")
    
    print("=" * 60)
    print(f"最终模型: k_value_mappo_final.pth")
    print("=" * 60)
    
    # 保存训练数据到文件（参照MADQN格式）
    training_data = {
        'loss_history': train_env.loss_history,
        'policy_loss_history': train_env.policy_loss_history,
        'value_loss_history': train_env.value_loss_history,
        'entropy_loss_history': train_env.entropy_loss_history,
        'ent_coef_history': train_env.ent_coef_history,
        'total_rewards': train_env.total_rewards,
        'total_performance_ratios': train_env.total_performance_ratios
    }
    
    data_filename = f"mappo_training_data_{n_task_vehicles}vehicles.pkl"
    with open(data_filename, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"\n训练数据已保存: {data_filename}")


if __name__ == "__main__":
    main()
