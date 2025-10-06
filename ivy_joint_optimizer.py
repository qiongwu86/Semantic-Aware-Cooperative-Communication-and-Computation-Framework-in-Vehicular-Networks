"""
基于Ivy算法的k值和lambda联合优化求解器
用于高速公路语义卸载场景的快速优化

作者: 语义卸载系统
版本: 1.0
日期: 2024

主要功能:
1. 联合优化k_v2v, k_v2i值（离散变量）和lambda分配比例（连续变量）
2. 基于Ivy算法的高效求解
3. 快速接口调用，适用于环境仿真
4. 集成HighwayEnvironment的必要函数
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
try:
    from HighwayEnvironment import HighwayEnvironment
    from lambda_lp_solver import compute_optimal_lambda
except ImportError as e:
    print(f"警告: 导入模块失败 {e}")
    print("请确保HighwayEnvironment.py和lambda_lp_solver.py在同一目录下")


class IvyJointOptimizer:
    """
    基于Ivy算法的k值和lambda联合优化器
    
    核心思想:
    - 外层: 使用Ivy算法优化离散的k_v2v, k_v2i值
    - 内层: 给定k值，使用LP求解最优lambda分配
    
    Ivy算法特点:
    - 模拟常春藤植物的生长和攀爬行为
    - 天然适合离散优化问题
    - 收敛速度快，参数少
    """
    
    def __init__(self, 
                 n_ivy: int = 20,
                 max_iter: int = 50,
                 growth_prob_init: float = 0.8,
                 growth_prob_final: float = 0.3,
                 random_seed: int = None):
        """
        初始化Ivy优化器
        
        Args:
            n_ivy: Ivy个体数量
            max_iter: 最大迭代次数
            growth_prob_init: 初始生长概率（高探索）
            growth_prob_final: 最终生长概率（高开发）
            random_seed: 随机种子，用于结果重现
        """
        self.n_ivy = n_ivy
        self.max_iter = max_iter
        self.growth_prob_init = growth_prob_init
        self.growth_prob_final = growth_prob_final
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # k值范围
        self.k_min = 1
        self.k_max = 20
        
        # 优化历史记录
        self.optimization_history = []
        self.convergence_history = []
        
        # 性能统计
        self.total_evaluations = 0
        self.optimization_time = 0
    
    def optimize(self, 
                 snr_v2v: float,
                 snr_v2i: float, 
                 task_size: float,
                 env: HighwayEnvironment,
                 service_vehicle_tasks: int = 1,
                 total_tasks_in_slot: int = 1,
                 verbose: bool = False) -> Tuple[Tuple[int, int], Tuple[float, float, float], float, Dict]:
        """
        联合优化k值和lambda值
        
        Args:
            snr_v2v: V2V链路信噪比 (dB)
            snr_v2i: V2I链路信噪比 (dB)
            task_size: 任务大小 (Mbit)
            env: 高速公路环境实例
            service_vehicle_tasks: 服务车辆处理的任务数
            total_tasks_in_slot: 当前时隙总任务数
            verbose: 是否输出详细信息
            
        Returns:
            optimal_k: (k_v2v, k_v2i) 最优k值组合
            optimal_lambda: (lambda_local, lambda_edge, lambda_bs) 最优lambda分配
            optimal_delay: 最优总延迟 (秒)
            optimization_info: 优化过程信息
        """
        start_time = time.time()
        self.total_evaluations = 0
        self.convergence_history = []
        
        if verbose:
            print(f"开始Ivy联合优化:")
            print(f"  SNR_V2V: {snr_v2v:.2f} dB, SNR_V2I: {snr_v2i:.2f} dB")
            print(f"  任务大小: {task_size:.3f} Mbit")
            print(f"  Ivy参数: n_ivy={self.n_ivy}, max_iter={self.max_iter}")
        
        # 初始化Ivy群体（直接在离散空间）
        ivy_population = self._initialize_ivy_population()
        
        # 全局最优解跟踪
        global_best_k = None
        global_best_lambda = None
        global_best_delay = float('inf')
        
        # 个体历史最优解
        personal_best_k = [None] * self.n_ivy
        personal_best_delay = [float('inf')] * self.n_ivy
        
        # Ivy进化循环
        for iteration in range(self.max_iter):
            iteration_best_delay = float('inf')
            current_fitness_list = []
            
            # 评估当前群体
            for i, (k_v2v, k_v2i) in enumerate(ivy_population):
                # 评估当前k值组合
                lambda_local, lambda_edge, lambda_bs, total_delay = self._evaluate_k_combination(
                    k_v2v, k_v2i, snr_v2v, snr_v2i, task_size, env, service_vehicle_tasks, total_tasks_in_slot
                )
                current_fitness_list.append(total_delay)
                
                # 更新个体最优
                if total_delay < personal_best_delay[i]:
                    personal_best_delay[i] = total_delay
                    personal_best_k[i] = (k_v2v, k_v2i)
                
                # 更新全局最优
                if total_delay < global_best_delay:
                    global_best_delay = total_delay
                    global_best_k = (k_v2v, k_v2i)
                    global_best_lambda = (lambda_local, lambda_edge, lambda_bs)
                    
                    if verbose and iteration % 10 == 0:
                        print(f"  迭代 {iteration}: 新最优解 k=({k_v2v},{k_v2i}), 延迟={total_delay:.6f}s")
                
                iteration_best_delay = min(iteration_best_delay, total_delay)
            
            # 记录收敛历史
            self.convergence_history.append(iteration_best_delay)
            
            # 早停检查
            if self._check_convergence(iteration):
                if verbose:
                    print(f"  提前收敛于迭代 {iteration}")
                break
            
            # Ivy生长更新
            ivy_population = self._ivy_growth_update(
                ivy_population, personal_best_k, global_best_k, 
                current_fitness_list, personal_best_delay, iteration
            )
        
        # 计算优化时间
        self.optimization_time = time.time() - start_time
        
        # 准备返回信息
        optimization_info = {
            'iterations': min(iteration + 1, self.max_iter),
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'convergence_history': self.convergence_history.copy(),
            'final_population_diversity': self._calculate_population_diversity(ivy_population),
            'algorithm': 'Ivy'
        }
        
        if verbose:
            print(f"优化完成!")
            print(f"  最优k值: k_v2v={global_best_k[0]}, k_v2i={global_best_k[1]}")
            print(f"  最优lambda: local={global_best_lambda[0]:.3f}, edge={global_best_lambda[1]:.3f}, bs={global_best_lambda[2]:.3f}")
            print(f"  最优延迟: {global_best_delay:.6f}s")
            print(f"  优化时间: {self.optimization_time:.3f}s")
            print(f"  总评估次数: {self.total_evaluations}")
        
        return global_best_k, global_best_lambda, global_best_delay, optimization_info
    
    def _initialize_ivy_population(self) -> List[Tuple[int, int]]:
        """初始化Ivy群体"""
        population = []
        for _ in range(self.n_ivy):
            k_v2v = np.random.randint(self.k_min, self.k_max + 1)
            k_v2i = np.random.randint(self.k_min, self.k_max + 1)
            population.append((k_v2v, k_v2i))
        return population


class IvyLambdaOptimizer:
    """
    基于Ivy思想的连续λ优化器（仅优化lambda，不搜索k）

    目标：
      给定三分支在满载情况下的延迟系数 A = [A_local, A_edge, A_bs]，
      最小化 T(λ) = max(λ_local*A_local, λ_edge*A_edge, λ_bs*A_bs)
      s.t. λ_i ≥ 0, Σλ_i = 1

    说明：
      - 问题是凸的且有闭式解（与LP一致），此优化器用于与LP形成对比的元启发式连续求解。
      - 支持两种参数化方法：softmax 和 单纯形投影
      - 实现Ivy算法的"向目标生长 + 随机探索"机制
    """

    def __init__(self,
                 n_ivy: int = 15,
                 max_iter: int = 50,
                 growth_prob_init: float = 0.8,
                 growth_prob_final: float = 0.3,
                 step_init: float = 0.3,
                 step_final: float = 0.05,
                 noise_std_init: float = 0.15,
                 noise_std_final: float = 0.02,
                 parametrization: str = "softmax",  # "softmax" or "simplex"
                 random_seed: int = None):
        """
        初始化连续λ优化器
        
        Args:
            n_ivy: Ivy个体数量
            max_iter: 最大迭代次数
            growth_prob_init/final: 生长概率（向全局最优靠近）的初始/最终值
            step_init/final: 生长步长的初始/最终值  
            noise_std_init/final: 随机探索噪声标准差的初始/最终值
            parametrization: 参数化方法 "softmax" 或 "simplex"
            random_seed: 随机种子
        """
        self.n_ivy = n_ivy
        self.max_iter = max_iter
        self.growth_prob_init = growth_prob_init
        self.growth_prob_final = growth_prob_final
        self.step_init = step_init
        self.step_final = step_final
        self.noise_std_init = noise_std_init
        self.noise_std_final = noise_std_final
        self.parametrization = parametrization

        if random_seed is not None:
            np.random.seed(random_seed)

        # 统计信息
        self.total_evaluations = 0
        self.optimization_time = 0.0
        self.convergence_history = []

    @staticmethod
    def _softmax(theta: np.ndarray) -> np.ndarray:
        """稳定的softmax实现"""
        if theta.ndim == 1:
            x = theta - np.max(theta)
            e = np.exp(x)
            return e / (np.sum(e) + 1e-12)
        else:  # batch
            x = theta - np.max(theta, axis=-1, keepdims=True)
            e = np.exp(x)
            return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

    @staticmethod
    def _project_to_simplex(x: np.ndarray) -> np.ndarray:
        """投影到单纯形：λ_i ≥ 0, Σλ_i = 1"""
        if x.ndim == 1:
            # 单个向量
            n = len(x)
            if np.sum(x) == 1 and np.all(x >= 0):
                return x
            
            # 排序投影算法
            u = np.sort(x)[::-1]  # 降序
            cssv = np.cumsum(u) - 1.0
            ind = np.arange(n) + 1
            cond = u - cssv / ind > 0
            rho = ind[cond][-1] if np.any(cond) else 1
            theta = cssv[rho - 1] / rho
            return np.maximum(x - theta, 0)
        else:
            # 批量投影
            return np.array([IvyLambdaOptimizer._project_to_simplex(row) for row in x])

    @staticmethod
    def _objective_T(lmbd: np.ndarray, A: np.ndarray) -> float:
        """目标函数：最大化三分支延迟"""
        return float(np.max(lmbd * A))

    def _get_schedule(self, iteration: int) -> Tuple[float, float, float]:
        """获取当前迭代的参数调度"""
        progress = iteration / max(1, self.max_iter - 1)
        
        growth_prob = self.growth_prob_init * (1 - progress) + self.growth_prob_final * progress
        step_size = self.step_init * (1 - progress) + self.step_final * progress
        noise_std = self.noise_std_init * (1 - progress) + self.noise_std_final * progress
        
        return growth_prob, step_size, noise_std

    def _initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """初始化群体，返回(λ群体, θ群体)"""
        if self.parametrization == "softmax":
            # 在θ空间初始化，然后映射到λ
            theta_pop = np.random.randn(self.n_ivy, 3) * 0.5
            lambda_pop = self._softmax(theta_pop)
            return lambda_pop, theta_pop
        else:  # simplex
            # 直接在单纯形上随机初始化
            raw = np.random.exponential(1.0, (self.n_ivy, 3))
            lambda_pop = raw / np.sum(raw, axis=1, keepdims=True)
            return lambda_pop, None  # 不需要θ空间

    def _growth_towards_target_theta(self, current_theta: np.ndarray, target_theta: np.ndarray, 
                                    step_size: float) -> np.ndarray:
        """向目标θ生长（softmax参数化）"""
        direction = target_theta - current_theta
        new_theta = current_theta + step_size * direction
        return new_theta

    def _random_exploration_theta(self, current_theta: np.ndarray, noise_std: float) -> np.ndarray:
        """随机探索θ空间（softmax参数化）"""
        noise = np.random.normal(0, noise_std, current_theta.shape)
        new_theta = current_theta + noise
        return new_theta

    def _growth_towards_target_lambda(self, current_lambda: np.ndarray, target_lambda: np.ndarray, 
                                     step_size: float) -> np.ndarray:
        """向目标λ生长（simplex参数化）"""
        direction = target_lambda - current_lambda
        new_lambda = current_lambda + step_size * direction
        new_lambda = self._project_to_simplex(new_lambda)
        return new_lambda

    def _random_exploration_lambda(self, current_lambda: np.ndarray, noise_std: float) -> np.ndarray:
        """随机探索λ空间（simplex参数化）"""
        noise = np.random.normal(0, noise_std, current_lambda.shape)
        new_lambda = current_lambda + noise
        new_lambda = self._project_to_simplex(new_lambda)
        return new_lambda

    def _check_convergence(self, iteration: int, tolerance: float = 1e-8) -> bool:
        """检查收敛性"""
        if iteration < 10:
            return False
        
        recent_history = self.convergence_history[-10:]
        improvement = max(recent_history) - min(recent_history)
        return improvement < tolerance

    @staticmethod
    def get_closed_form_solution(local_A: float, edge_A: float, bs_A: float) -> Tuple[float, float, float, float]:
        """
        获取闭式解（与LP一致），用于对比验证
        
        理论解：λ_i = (1/A_i) / Σ(1/A_j), T* = 1 / Σ(1/A_j)
        
        Returns:
            (lambda_local, lambda_edge, lambda_bs, optimal_T)
        """
        A = np.array([local_A, edge_A, bs_A], dtype=np.float64)
        # 处理数值稳定性
        A = np.maximum(A, 1e-12)
        
        inv_A = 1.0 / A
        sum_inv_A = np.sum(inv_A)
        
        lambda_optimal = inv_A / sum_inv_A
        T_optimal = 1.0 / sum_inv_A
        
        return float(lambda_optimal[0]), float(lambda_optimal[1]), float(lambda_optimal[2]), float(T_optimal)

    def optimize_lambda(self,
                       local_A: float,
                       edge_A: float,
                       bs_A: float,
                       verbose: bool = False) -> Tuple[float, float, float, float, Dict]:
        """
        优化λ分配比例
        
        Args:
            local_A: 本地分支满载延迟（秒）
            edge_A: 边缘分支满载延迟（传输+服务计算，秒）
            bs_A: 基站分支满载延迟（传输+基站计算，秒）
            verbose: 是否输出详细信息
            
        Returns:
            (lambda_local, lambda_edge, lambda_bs, optimal_T, optimization_info)
        """
        import time
        
        start_time = time.time()
        self.total_evaluations = 0
        self.convergence_history = []
        
        if verbose:
            print(f"开始Ivy连续λ优化:")
            print(f"  分支延迟系数 A: local={local_A:.6f}, edge={edge_A:.6f}, bs={bs_A:.6f}")
            print(f"  参数化方法: {self.parametrization}")
            print(f"  Ivy参数: n_ivy={self.n_ivy}, max_iter={self.max_iter}")
        
        A = np.array([local_A, edge_A, bs_A], dtype=np.float64)
        
        # 初始化群体
        lambda_population, theta_population = self._initialize_population()
        
        # 评估初始群体
        fitness = np.array([self._objective_T(lmbd, A) for lmbd in lambda_population])
        self.total_evaluations += self.n_ivy
        
        # 全局最优跟踪
        global_best_idx = np.argmin(fitness)
        global_best_lambda = lambda_population[global_best_idx].copy()
        global_best_T = float(fitness[global_best_idx])
        
        if self.parametrization == "softmax":
            global_best_theta = theta_population[global_best_idx].copy()
            personal_best_theta = theta_population.copy()
        else:
            global_best_theta = None
            personal_best_theta = None
        
        # 个体历史最优
        personal_best_lambda = lambda_population.copy()
        personal_best_T = fitness.copy()
        
        # Ivy进化循环
        for iteration in range(self.max_iter):
            iteration_best_T = float(np.min(fitness))
            self.convergence_history.append(iteration_best_T)
            
            # 早停检查
            if self._check_convergence(iteration):
                if verbose:
                    print(f"  提前收敛于迭代 {iteration}")
                break
            
            # 获取当前参数
            growth_prob, step_size, noise_std = self._get_schedule(iteration)
            
            # 更新每个个体
            new_lambda_population = []
            new_theta_population = [] if self.parametrization == "softmax" else None
            
            for i in range(self.n_ivy):
                current_lambda = lambda_population[i]
                
                # 自适应生长概率（性能差的个体更倾向于向全局最优靠近）
                adaptive_growth_prob = growth_prob
                if personal_best_T[i] > global_best_T:
                    fitness_ratio = personal_best_T[i] / (global_best_T + 1e-12)
                    adaptive_growth_prob = min(0.95, growth_prob * fitness_ratio)
                
                # 根据参数化方法选择操作空间
                if self.parametrization == "softmax":
                    current_theta = theta_population[i]
                    
                    # 在θ空间进行操作
                    if np.random.random() < adaptive_growth_prob:
                        # 向全局最优生长
                        new_theta = self._growth_towards_target_theta(current_theta, global_best_theta, step_size)
                    else:
                        # 随机探索
                        new_theta = self._random_exploration_theta(current_theta, noise_std)
                    
                    # 映射到λ空间
                    new_lambda = self._softmax(new_theta)
                    new_theta_population.append(new_theta)
                else:  # simplex
                    # 直接在λ空间操作
                    if np.random.random() < adaptive_growth_prob:
                        # 向全局最优生长
                        new_lambda = self._growth_towards_target_lambda(current_lambda, global_best_lambda, step_size)
                    else:
                        # 随机探索
                        new_lambda = self._random_exploration_lambda(current_lambda, noise_std)
                
                new_lambda_population.append(new_lambda)
            
            new_lambda_population = np.array(new_lambda_population)
            if self.parametrization == "softmax":
                new_theta_population = np.array(new_theta_population)
            
            # 评估新群体
            new_fitness = np.array([self._objective_T(lmbd, A) for lmbd in new_lambda_population])
            self.total_evaluations += self.n_ivy
            
            # 贪心选择（只有更好的解才被接受）
            improved = new_fitness < fitness
            lambda_population[improved] = new_lambda_population[improved]
            fitness[improved] = new_fitness[improved]
            
            if self.parametrization == "softmax":
                theta_population[improved] = new_theta_population[improved]
            
            # 更新个体最优
            better_personal = fitness < personal_best_T
            personal_best_T[better_personal] = fitness[better_personal]
            personal_best_lambda[better_personal] = lambda_population[better_personal]
            
            if self.parametrization == "softmax":
                personal_best_theta[better_personal] = theta_population[better_personal]
            
            # 更新全局最优
            current_best_idx = np.argmin(fitness)
            current_best_T = float(fitness[current_best_idx])
            if current_best_T < global_best_T:
                global_best_T = current_best_T
                global_best_lambda = lambda_population[current_best_idx].copy()
                
                if self.parametrization == "softmax":
                    global_best_theta = theta_population[current_best_idx].copy()
                
                if verbose and iteration % 10 == 0:
                    print(f"  迭代 {iteration}: 新最优解 T={global_best_T:.8f}")
        
        self.optimization_time = time.time() - start_time
        
        # 准备返回信息
        optimization_info = {
            'iterations': len(self.convergence_history),
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'convergence_history': self.convergence_history.copy(),
            'parametrization': self.parametrization,
            'algorithm': 'Ivy-Lambda-Continuous'
        }
        
        if verbose:
            print(f"优化完成!")
            print(f"  最优λ: local={global_best_lambda[0]:.6f}, edge={global_best_lambda[1]:.6f}, bs={global_best_lambda[2]:.6f}")
            print(f"  最优延迟: {global_best_T:.8f}s")
            print(f"  优化时间: {self.optimization_time:.4f}s")
            print(f"  总评估次数: {self.total_evaluations}")
            
            # 与闭式解对比
            cf_l, cf_e, cf_b, cf_T = self.get_closed_form_solution(local_A, edge_A, bs_A)
            error = abs(global_best_T - cf_T) / (cf_T + 1e-12) * 100
            print(f"  闭式解对比: T_closed={cf_T:.8f}, 误差={error:.4f}%")
        
        return (float(global_best_lambda[0]), 
                float(global_best_lambda[1]), 
                float(global_best_lambda[2]), 
                float(global_best_T), 
                optimization_info)
    
    def _evaluate_k_combination(self, 
                               k_v2v: int, 
                               k_v2i: int, 
                               snr_v2v: float, 
                               snr_v2i: float, 
                               task_size: float, 
                               env: HighwayEnvironment,
                               service_vehicle_tasks: int = 1,
                               total_tasks_in_slot: int = 1) -> Tuple[float, float, float, float]:
        """
        评估给定k值组合的性能（内层LP优化）
        
        Args:
            service_vehicle_tasks: 服务车辆处理的任务数
            total_tasks_in_slot: 当前时隙总任务数
        
        Returns:
            lambda_local, lambda_edge, lambda_bs, total_delay
        """
        self.total_evaluations += 1
        
        try:
            # 计算各分支的单位延迟（满载时的延迟）
            
            # 1. 本地计算延迟
            local_A = env.calculate_computation_delay(task_size, env.local_cpu_freq)
            
            # 2. 边缘分支延迟 = V2V传输 + 服务车辆计算
            _, v2v_transmission_delay = env.calculate_semantic_rate_and_delay(
                snr_v2v, k_v2v, task_size
            )
            service_computation_delay = env.calculate_computation_delay(
                task_size, env.service_cpu_freq, service_vehicle_tasks  # 考虑实际并发任务数
            )
            edge_A = v2v_transmission_delay + service_computation_delay
            
            # 3. 基站分支延迟 = V2I传输 + 基站计算
            _, v2i_transmission_delay = env.calculate_semantic_rate_and_delay(
                snr_v2i, k_v2i, task_size
            )
            bs_computation_delay = env.calculate_computation_delay(
                task_size, env.bs_cpu_freq, total_tasks_in_slot  # 考虑实际并发任务数
            )
            bs_A = v2i_transmission_delay + bs_computation_delay
            
            # 4. 使用LP求解最优lambda分配
            lambda_local, lambda_edge, lambda_bs, optimal_delay = compute_optimal_lambda(
                local_A, edge_A, bs_A, prefer="closed_form"
            )
            
            return lambda_local, lambda_edge, lambda_bs, optimal_delay
            
        except Exception as e:
            # 异常情况返回极大值
            print(f"警告: 评估k=({k_v2v},{k_v2i})时出错: {e}")
            return 0.33, 0.33, 0.34, float('inf')
    
    def _ivy_growth_update(self, 
                          population: List[Tuple[int, int]],
                          personal_best_k: List[Tuple[int, int]],
                          global_best_k: Tuple[int, int],
                          current_fitness: List[float],
                          personal_best_fitness: List[float],
                          iteration: int) -> List[Tuple[int, int]]:
        """Ivy生长更新机制"""
        
        new_population = []
        
        for i in range(self.n_ivy):
            current_k = population[i]
            
            # 自适应生长概率
            progress = iteration / self.max_iter
            growth_prob = self.growth_prob_init * (1 - progress) + self.growth_prob_final * progress
            
            # 基于适应度调整生长概率
            if personal_best_fitness[i] < float('inf'):
                fitness_ratio = current_fitness[i] / (personal_best_fitness[i] + 1e-8)
                growth_prob *= min(2.0, fitness_ratio)  # 性能差的个体更倾向于生长
            
            # Ivy生长决策
            if np.random.random() < growth_prob and global_best_k is not None:
                # 向最优解生长
                new_k = self._grow_towards_target(current_k, global_best_k, iteration)
            else:
                # 随机探索（保持多样性）
                new_k = self._random_exploration(current_k)
            
            new_population.append(new_k)
        
        return new_population
    
    def _grow_towards_target(self, 
                           current_k: Tuple[int, int], 
                           target_k: Tuple[int, int], 
                           iteration: int) -> Tuple[int, int]:
        """向目标k值生长"""
        
        current_k_v2v, current_k_v2i = current_k
        target_k_v2v, target_k_v2i = target_k
        
        # 自适应步长（早期大步长，后期小步长）
        max_step = max(1, int(3 * (1 - iteration / self.max_iter)))
        
        # V2V维度生长
        new_k_v2v = self._grow_single_dimension(current_k_v2v, target_k_v2v, max_step)
        
        # V2I维度生长
        new_k_v2i = self._grow_single_dimension(current_k_v2i, target_k_v2i, max_step)
        
        return (new_k_v2v, new_k_v2i)
    
    def _grow_single_dimension(self, current: int, target: int, max_step: int) -> int:
        """单维度生长"""
        if current == target:
            # 已到达目标，在附近小幅扰动
            perturbation = np.random.randint(-1, 2)  # -1, 0, 1
            return np.clip(current + perturbation, self.k_min, self.k_max)
        else:
            # 向目标移动
            direction = 1 if target > current else -1
            step_size = min(max_step, abs(target - current))
            actual_step = np.random.randint(1, step_size + 1)
            new_value = current + direction * actual_step
            return np.clip(new_value, self.k_min, self.k_max)
    
    def _random_exploration(self, current_k: Tuple[int, int]) -> Tuple[int, int]:
        """随机探索机制"""
        current_k_v2v, current_k_v2i = current_k
        
        # 在当前位置附近随机探索
        exploration_radius = 3
        
        new_k_v2v = current_k_v2v + np.random.randint(-exploration_radius, exploration_radius + 1)
        new_k_v2i = current_k_v2i + np.random.randint(-exploration_radius, exploration_radius + 1)
        
        # 边界约束
        new_k_v2v = np.clip(new_k_v2v, self.k_min, self.k_max)
        new_k_v2i = np.clip(new_k_v2i, self.k_min, self.k_max)
        
        return (new_k_v2v, new_k_v2i)
    
    def _check_convergence(self, iteration: int) -> bool:
        """检查收敛性"""
        if iteration < 10:  # 至少运行10次迭代
            return False
        
        # 检查最近几次迭代的改进
        recent_history = self.convergence_history[-10:]
        improvement = max(recent_history) - min(recent_history)
        
        # 如果改进很小，认为已收敛
        return improvement < 1e-8
    
    def _calculate_population_diversity(self, population: List[Tuple[int, int]]) -> float:
        """计算群体多样性"""
        if len(population) <= 1:
            return 0.0
        
        unique_individuals = set(population)
        diversity = len(unique_individuals) / len(population)
        return diversity


class FastIvyOptimizer:
    """
    快速Ivy优化器 - 专为环境仿真设计的轻量版本
    
    特点:
    - 更少的参数和迭代次数
    - 更快的收敛速度
    - 简化的接口
    """
    
    def __init__(self, n_ivy: int = 10, max_iter: int = 20):
        self.optimizer = IvyJointOptimizer(n_ivy=n_ivy, max_iter=max_iter)
    
    def quick_optimize(self, 
                      snr_v2v: float, 
                      snr_v2i: float, 
                      task_size: float, 
                      env: HighwayEnvironment,
                      service_vehicle_tasks: int = 1,
                      total_tasks_in_slot: int = 1) -> Tuple[Tuple[int, int], Tuple[float, float, float], float]:
        """
        快速优化接口
        
        Returns:
            optimal_k, optimal_lambda, optimal_delay
        """
        optimal_k, optimal_lambda, optimal_delay, _ = self.optimizer.optimize(
            snr_v2v, snr_v2i, task_size, env, service_vehicle_tasks, total_tasks_in_slot, verbose=False
        )
        return optimal_k, optimal_lambda, optimal_delay


def demo_ivy_lambda_optimizer():
    """演示Ivy连续λ优化器的使用"""
    print("=" * 70)
    print("IvyLambdaOptimizer 连续λ优化器演示")
    print("=" * 70)
    
    # 测试场景
    test_scenarios = [
        {"local_A": 0.12, "edge_A": 0.20, "bs_A": 0.15, "name": "典型场景"},
        {"local_A": 0.05, "edge_A": 0.50, "bs_A": 0.08, "name": "极端延迟比例"},
        {"local_A": 0.18, "edge_A": 0.15, "bs_A": 0.25, "name": "高延迟场景"},
        {"local_A": 0.08, "edge_A": 0.12, "bs_A": 0.30, "name": "边缘劣势场景"},
    ]
    
    print(f"\n🔵 测试 {len(test_scenarios)} 个延迟场景")
    print("-" * 70)
    
    total_errors_simplex = []
    total_errors_softmax = []
    total_times_simplex = []
    total_times_softmax = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- 场景 {i}: {scenario['name']} ---")
        local_A, edge_A, bs_A = scenario['local_A'], scenario['edge_A'], scenario['bs_A']
        print(f"延迟系数: local={local_A:.3f}, edge={edge_A:.3f}, bs={bs_A:.3f}")
        
        # 理论最优解（LP闭式解）
        cf_l, cf_e, cf_b, cf_T = IvyLambdaOptimizer.get_closed_form_solution(local_A, edge_A, bs_A)
        print(f"理论最优: λ=({cf_l:.4f}, {cf_e:.4f}, {cf_b:.4f}), T={cf_T:.6f}s")
        
        # 测试两种参数化方法
        for param_method in ["simplex", "softmax"]:
            optimizer = IvyLambdaOptimizer(
                n_ivy=15, max_iter=50, parametrization=param_method, random_seed=42+i
            )
            
            start_time = time.time()
            lam_l, lam_e, lam_b, T_ivy, info = optimizer.optimize_lambda(
                local_A, edge_A, bs_A, verbose=False
            )
            opt_time = time.time() - start_time
            
            error_percent = abs(T_ivy - cf_T) / cf_T * 100
            lambda_error = np.sqrt((lam_l - cf_l)**2 + (lam_e - cf_e)**2 + (lam_b - cf_b)**2)
            
            # 记录统计数据
            if param_method == "simplex":
                total_errors_simplex.append(error_percent)
                total_times_simplex.append(opt_time)
            else:
                total_errors_softmax.append(error_percent)
                total_times_softmax.append(opt_time)
            
            print(f"{param_method:>7}: λ=({lam_l:.4f}, {lam_e:.4f}, {lam_b:.4f}), T={T_ivy:.6f}s")
            print(f"{'':>9} T误差={error_percent:.3f}%, λ误差={lambda_error:.6f}")
            print(f"{'':>9} 时间={opt_time:.3f}s, 迭代={info['iterations']}, 评估={info['total_evaluations']}")
    
    print(f"\n🔵 性能统计总结")
    print("-" * 70)
    
    # Simplex 统计
    simplex_avg_error = np.mean(total_errors_simplex)
    simplex_max_error = np.max(total_errors_simplex)
    simplex_avg_time = np.mean(total_times_simplex)
    
    print(f"Simplex参数化:")
    print(f"  平均T误差: {simplex_avg_error:.3f}%, 最大T误差: {simplex_max_error:.3f}%")
    print(f"  平均优化时间: {simplex_avg_time:.3f}s")
    
    # Softmax 统计
    softmax_avg_error = np.mean(total_errors_softmax)
    softmax_max_error = np.max(total_errors_softmax)
    softmax_avg_time = np.mean(total_times_softmax)
    
    print(f"Softmax参数化:")
    print(f"  平均T误差: {softmax_avg_error:.3f}%, 最大T误差: {softmax_max_error:.3f}%")
    print(f"  平均优化时间: {softmax_avg_time:.3f}s")
    
    # 推荐
    if simplex_avg_error < softmax_avg_error:
        print(f"\n✅ 推荐使用: Simplex参数化 (更稳定，平均误差更小)")
    else:
        print(f"\n✅ 推荐使用: Softmax参数化 (更稳定，平均误差更小)")
    
    print(f"\n🔧 使用方式:")
    print(f"```python")
    print(f"from ivy_joint_optimizer import IvyLambdaOptimizer")
    print(f"")
    print(f"optimizer = IvyLambdaOptimizer(parametrization='simplex')")
    print(f"lambda_l, lambda_e, lambda_bs, T_opt, info = optimizer.optimize_lambda(")
    print(f"    local_A=0.12, edge_A=0.20, bs_A=0.15)")
    print(f"```")


if __name__ == "__main__":
    # 运行连续λ优化器演示
    demo_ivy_lambda_optimizer()
