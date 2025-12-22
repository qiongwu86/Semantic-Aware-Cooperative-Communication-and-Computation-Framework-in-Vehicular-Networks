"""
优化的测试主文件：一次运行环境，计算所有方案的性能对比
避免重复的车辆运动仿真，提高效率
"""

import os
# 设置环境变量解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import sys

# 条件导入torch - 绘图模式不需要torch
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        print("⚠️ 注意: 绘图模式运行中，跳过torch导入")
        torch = None
    else:
        print("❌ 错误: torch未安装，请安装torch或使用绘图模式")
        sys.exit(1)

# 条件导入依赖torch的模块
if TORCH_AVAILABLE:
    from HighwayEnvironment import HighwayEnvironment
    from k_value_mappo import KValueMAPPO
    from k_value_madqn import KValueMADQN
    from lambda_lp_solver import compute_optimal_lambda
    
    # 尝试导入IvyLambdaOptimizer，如果失败则跳过
    try:
        from ivy_joint_optimizer import IvyLambdaOptimizer
        IVY_AVAILABLE = True
    except ImportError:
        print("⚠️ 警告: ivy_joint_optimizer导入失败，Ivy方案将被跳过")
        IvyLambdaOptimizer = None
        IVY_AVAILABLE = False
else:
    # 绘图模式下的空导入
    HighwayEnvironment = None
    KValueMAPPO = None
    KValueMADQN = None
    compute_optimal_lambda = None
    IvyLambdaOptimizer = None
    IVY_AVAILABLE = False

# 条件导入MAPPO-SNR训练器
if TORCH_AVAILABLE:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # 文件名包含空格和连字符，需要动态导入
    import importlib.util
    mappo_snr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k_value_mappo -SNR.py")
    spec = importlib.util.spec_from_file_location("k_value_mappo_snr", mappo_snr_path)
    mappo_snr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mappo_snr_module)
    KValueMAPPO_SNR = mappo_snr_module.KValueMAPPO
else:
    # 绘图模式下的空导入
    import importlib.util
    KValueMAPPO_SNR = None


class OptimizedPerformanceComparator:
    """优化的性能对比测试器 - 一次仿真，多种方案对比"""
    
    def __init__(self, n_task_vehicles=20, n_service_vehicles=5, 
                 mappo_model_path="k_value_mappo_final.pth",
                 mappo_snr_model_path="mappo-snr/k_value_mappo_snr_final.pth",
                 madqn_model_path=None, task_size=0.8):
        """
        初始化测试环境
        Args:
            n_task_vehicles: 任务车辆数
            n_service_vehicles: 服务车辆数 
            mappo_model_path: 训练好的MAPPO模型路径
            mappo_snr_model_path: 训练好的MAPPO-SNR模型路径
            madqn_model_path: 训练好的MADQN模型路径
            task_size: 固定任务大小（Mbit）
        """
        self.n_task_vehicles = n_task_vehicles
        self.n_service_vehicles = n_service_vehicles
        self.task_size = task_size
        
        # 加载训练好的MAPPO模型
        self.mappo = KValueMAPPO()
        if os.path.exists(mappo_model_path):
            self.mappo.policy.load_state_dict(torch.load(mappo_model_path, map_location='cpu'))
            print(f"✅ 成功加载MAPPO模型: {mappo_model_path}")
        else:
            print(f"⚠️  未找到MAPPO模型文件: {mappo_model_path}，将使用随机初始化的模型")
        
        self.mappo.policy.eval()  # 设置为评估模式
        
        # 加载训练好的MAPPO-SNR模型
        self.mappo_snr = KValueMAPPO_SNR(enable_snr=True)
        if os.path.exists(mappo_snr_model_path):
            self.mappo_snr.policy.load_state_dict(torch.load(mappo_snr_model_path, map_location='cpu'))
            print(f"✅ 成功加载MAPPO-SNR模型: {mappo_snr_model_path}")
        else:
            print(f"⚠️  未找到MAPPO-SNR模型文件: {mappo_snr_model_path}，将使用随机初始化的模型")
        
        self.mappo_snr.policy.eval()  # 设置为评估模式
        
        # 初始化Ivy连续λ优化器
        self.ivy_lambda_optimizer = IvyLambdaOptimizer(
            n_ivy=15, 
            max_iter=20, 
            parametrization="softmax",
            random_seed=42
        )
        print(f"✅ Ivy连续λ优化器已初始化")
        
        # 加载训练好的MADQN模型
        self.madqn = KValueMADQN()
        if madqn_model_path and os.path.exists(madqn_model_path):
            try:
                # 加载完整的模型状态
                checkpoint = torch.load(madqn_model_path, map_location='cpu')
                
                # 检查是否是完整的训练状态（包含多个组件）
                if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                    # 这是完整的训练状态，只提取Q网络权重
                    self.madqn.q_network.load_state_dict(checkpoint['q_network'])
                    print(f"✅ 成功加载MADQN模型（从训练状态）: {madqn_model_path}")
                else:
                    # 这是单独的网络权重
                    self.madqn.q_network.load_state_dict(checkpoint)
                    print(f"✅ 成功加载MADQN模型（网络权重）: {madqn_model_path}")
            except Exception as e:
                print(f"⚠️  加载MADQN模型失败: {e}")
                print("将使用随机初始化的模型")
        elif madqn_model_path:
            print(f"⚠️  未找到MADQN模型文件: {madqn_model_path}，将使用随机初始化的模型")
        else:
            # 尝试自动查找MADQN模型
            auto_madqn_path = f"k_value_madqn_final_{n_task_vehicles}vehicles_per_exponential.pth"
            if os.path.exists(auto_madqn_path):
                try:
                    checkpoint = torch.load(auto_madqn_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                        self.madqn.q_network.load_state_dict(checkpoint['q_network'])
                    else:
                        self.madqn.q_network.load_state_dict(checkpoint)
                    print(f"✅ 自动加载MADQN模型: {auto_madqn_path}")
                except Exception as e:
                    print(f"⚠️  自动加载MADQN模型失败: {e}")
                    print("将使用随机初始化的模型")
            else:
                print(f"⚠️  未找到MADQN模型，将使用随机初始化的模型")
        
        self.madqn.q_network.eval()  # 设置为评估模式
        
        # 结果存储
        self.results = {
            'mappo_snr_lp': [],  # 新增MAPPO-SNR方案，放在首位
            'mappo_lp': [],
            'madqn_lp': [],
            'semantic_ivy': [],  # 语义-ivy方案
            'semantic_no_lp': [],
            'shannon': [],
            'shannon_ivy': [],  # 香农-ivy方案
            'shannon_no_lp': [],
            'traditional_k': []
        }
        
        print(f"🚗 优化测试环境初始化完成：{n_task_vehicles}辆任务车，{n_service_vehicles}辆服务车")
        print(f"📦 固定任务大小：{task_size} Mbit")
    
    def get_traditional_k(self, snr_dB):
        """传统的k值选择规则"""
        if snr_dB < -10:
            return 20
        elif snr_dB > 20:
            return 1
        else:
            # 简单的线性映射：SNR越高，k越小
            k = int(20 - (snr_dB + 10) * 19 / 30)
            return max(1, min(20, k))
    
    def get_mappo_k_value(self, snr_dB, task_size, semantic_table, link_type='v2v', distance=0.5):
        """使用MAPPO选择k值"""
        if not (-10 <= snr_dB <= 20):
            # 超出训练范围，使用传统规则
            return self.get_traditional_k(snr_dB)
        
        # 链路类型转换：V2V=0, V2I=1
        link_type_int = 0 if link_type == 'v2v' else 1
        
        # 构建观测（使用实际距离）
        obs = self.mappo.build_observation(
            snr_dB=snr_dB,
            distance=distance,  # 使用实际归一化距离
            task_size=task_size,
            link_type=link_type_int,  # 使用整数链路类型
            semantic_table=semantic_table
        )
        
        # 构建动作掩码
        action_mask = self.mappo.build_action_mask(snr_dB)
        
        # 选择最佳动作（确定性）
        k_value = self.mappo.select_best_action(obs, action_mask)
        
        return k_value
    
    def get_mappo_snr_k_value(self, snr_dB, task_size, semantic_table, link_type='v2v', distance=0.5):
        """使用MAPPO-SNR选择k值"""
        if not (-10 <= snr_dB <= 20):
            # 超出训练范围，使用传统规则
            return self.get_traditional_k(snr_dB)
        
        # 链路类型转换：V2V=0, V2I=1
        link_type_int = 0 if link_type == 'v2v' else 1
        
        # 构建观测（使用实际距离）
        obs = self.mappo_snr.build_observation(
            snr_dB=snr_dB,
            distance=distance,  # 使用实际归一化距离
            task_size=task_size,
            link_type=link_type_int,  # 使用整数链路类型
            semantic_table=semantic_table
        )
        
        # 构建动作掩码
        action_mask = self.mappo_snr.build_action_mask(snr_dB)
        
        # 选择最佳动作（确定性）
        k_value = self.mappo_snr.select_best_action(obs, action_mask)
        
        return k_value
    
    def get_madqn_k_value(self, snr_dB, task_size, semantic_table, link_type, distance=0.5):
        """
        使用MADQN选择k值
        Args:
            snr_dB: SNR值 (dB)
            task_size: 任务大小 (MB)
            semantic_table: 语义表
            link_type: 链路类型 ('v2v' 或 'v2i')
            distance: 归一化距离 (0-1)
        Returns:
            int: 选择的k值 (1-20)
        """
        # 检查SNR是否在训练范围内
        if snr_dB < -10 or snr_dB > 20:
            # 超出训练范围，使用传统规则
            return self.get_traditional_k(snr_dB)
        
        # 链路类型转换：V2V=0, V2I=1
        link_type_int = 0 if link_type == 'v2v' else 1
        
        # 构建观测
        obs = self.madqn.build_observation(
            snr_dB=snr_dB,
            distance=distance,  # 使用实际归一化距离
            task_size=task_size,
            link_type=link_type_int,
            semantic_table=semantic_table
        )
        
        # 构建动作掩码
        action_mask = self.madqn.build_action_mask(snr_dB)
        
        # 使用MADQN的测试阶段动作选择
        action_idx, _ = self.madqn.select_action(obs, action_mask, training=False)
        k_value = action_idx.item() + 1  # 转换为1-20的k值
        
        return k_value
    
    def calculate_delays_for_all_schemes(self, snr_v2v, snr_v2i, task_size, env, 
                                        service_vehicle_tasks=1, total_tasks_in_slot=1,
                                        task_vehicle_id=None, service_vehicle_id=None):
        """
        基于给定的SNR数据，计算所有方案的延迟
        Args:
            snr_v2v: V2V链路SNR
            snr_v2i: V2I链路SNR  
            task_size: 任务大小
            env: 环境实例
            service_vehicle_tasks: 该服务车辆在当前时隙处理的任务数
            total_tasks_in_slot: 当前时隙总任务数（基站需要处理的数量）
            task_vehicle_id: 任务车辆ID
            service_vehicle_id: 服务车辆ID
        Returns:
            dict: 包含所有方案延迟的字典
        """
        results = {}
        
        # 计算实际距离（用于MAPPO和MADQN的观测构建）
        # 需要分别计算V2V和V2I距离，因为它们的计算方式不同
        v2v_distance = 0.5  # V2V链路的归一化距离
        v2i_distance = 0.5  # V2I链路的归一化距离
        
        if task_vehicle_id is not None and service_vehicle_id is not None:
            try:
                # 找到对应的车辆
                task_vehicle = next(v for v in env.task_vehicles if v.id == task_vehicle_id)
                service_vehicle = next(v for v in env.service_vehicles if v.id == service_vehicle_id)
                
                import math
                
                # V2V距离：任务车辆到服务车辆的距离
                v2v_distance_meters = math.hypot(
                    task_vehicle.position[0] - service_vehicle.position[0],
                    task_vehicle.position[1] - service_vehicle.position[1]
                )
                # 归一化距离，参考MAPPO中的400米最大距离
                v2v_distance = min(v2v_distance_meters / 400.0, 1.0)
                
                # V2I距离：任务车辆到基站的距离（基站在道路中心）
                bs_pos = [env.width/2, env.height/2]
                v2i_distance_meters = math.hypot(
                    task_vehicle.position[0] - bs_pos[0],
                    task_vehicle.position[1] - bs_pos[1]
                )
                # 归一化距离，参考MAPPO中的400米最大距离
                v2i_distance = min(v2i_distance_meters / 400.0, 1.0)
                
            except (StopIteration, AttributeError):
                # 如果找不到车辆或无位置信息，使用默认距离
                v2v_distance = 0.5
                v2i_distance = 0.5
        
        # ================== 方案1: MAPPO-SNR + LP ==================
        mappo_snr_k_v2v = self.get_mappo_snr_k_value(snr_v2v, task_size, env.semantic_table, 'v2v', v2v_distance)
        mappo_snr_k_v2i = self.get_mappo_snr_k_value(snr_v2i, task_size, env.semantic_table, 'v2i', v2i_distance)
        
        # 计算各分支的完整任务延迟（用于LP）
        local_delay = env.calculate_computation_delay(task_size, env.local_cpu_freq)
        
        # V2V + 服务车辆分支
        _, v2v_snr_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2v, mappo_snr_k_v2v, task_size)
        service_comp_delay = env.calculate_computation_delay(task_size, env.service_cpu_freq, service_vehicle_tasks)
        edge_snr_delay = v2v_snr_sem_delay + service_comp_delay
        
        # V2I + 基站分支
        _, v2i_snr_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2i, mappo_snr_k_v2i, task_size)
        bs_comp_delay = env.calculate_computation_delay(task_size, env.bs_cpu_freq, total_tasks_in_slot)
        bs_snr_delay = v2i_snr_sem_delay + bs_comp_delay
        
        # 使用LP求解最优lambda
        lambda_snr_local, lambda_snr_edge, lambda_snr_bs, mappo_snr_lp_delay = compute_optimal_lambda(
            local_delay, edge_snr_delay, bs_snr_delay
        )
        
        results['mappo_snr_lp'] = {
            'total_delay_semantic': mappo_snr_lp_delay,
            'k_v2v': mappo_snr_k_v2v,
            'k_v2i': mappo_snr_k_v2i,
            'lambda_local': lambda_snr_local,
            'lambda_edge': lambda_snr_edge,
            'lambda_bs': lambda_snr_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_snr_sem_delay,
            'service_computation_delay': service_comp_delay,
            'v2i_transmission_delay': v2i_snr_sem_delay,
            'bs_computation_delay': bs_comp_delay
        }
        
        # ================== 方案2: MAPPO + LP ==================
        mappo_k_v2v = self.get_mappo_k_value(snr_v2v, task_size, env.semantic_table, 'v2v', v2v_distance)
        mappo_k_v2i = self.get_mappo_k_value(snr_v2i, task_size, env.semantic_table, 'v2i', v2i_distance)
        
        # 计算各分支的完整任务延迟（用于LP）
        # local_delay已在上面计算过了
        
        # V2V + 服务车辆分支
        _, v2v_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2v, mappo_k_v2v, task_size)
        # service_comp_delay已在上面计算过了
        edge_delay = v2v_sem_delay + service_comp_delay
        
        # V2I + 基站分支
        _, v2i_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2i, mappo_k_v2i, task_size)
        # bs_comp_delay已在上面计算过了
        bs_delay = v2i_sem_delay + bs_comp_delay
        
        # 使用LP求解最优lambda
        lambda_local, lambda_edge, lambda_bs, mappo_lp_delay = compute_optimal_lambda(
            local_delay, edge_delay, bs_delay
        )
        
        results['mappo_lp'] = {
            'total_delay_semantic': mappo_lp_delay,
            'k_v2v': mappo_k_v2v,
            'k_v2i': mappo_k_v2i,
            'lambda_local': lambda_local,
            'lambda_edge': lambda_edge,
            'lambda_bs': lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_sem_delay,
            'service_computation_delay': service_comp_delay,
            'v2i_transmission_delay': v2i_sem_delay,
            'bs_computation_delay': bs_comp_delay
        }
        
        # ================== 方案2: MADQN + LP ==================
        madqn_k_v2v = self.get_madqn_k_value(snr_v2v, task_size, env.semantic_table, 'v2v', v2v_distance)
        madqn_k_v2i = self.get_madqn_k_value(snr_v2i, task_size, env.semantic_table, 'v2i', v2i_distance)
        
        # 计算各分支的完整任务延迟（用于LP）
        madqn_local_delay = local_delay  # 本地计算不变
        
        # V2V + 服务车辆分支
        _, madqn_v2v_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2v, madqn_k_v2v, task_size)
        madqn_service_comp_delay = env.calculate_computation_delay(task_size, env.service_cpu_freq, service_vehicle_tasks)
        madqn_edge_delay = madqn_v2v_sem_delay + madqn_service_comp_delay
        
        # V2I + 基站分支
        _, madqn_v2i_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2i, madqn_k_v2i, task_size)
        madqn_bs_comp_delay = env.calculate_computation_delay(task_size, env.bs_cpu_freq, total_tasks_in_slot)
        madqn_bs_delay = madqn_v2i_sem_delay + madqn_bs_comp_delay
        
        # 使用LP求解最优lambda
        madqn_lambda_local, madqn_lambda_edge, madqn_lambda_bs, madqn_lp_delay = compute_optimal_lambda(
            madqn_local_delay, madqn_edge_delay, madqn_bs_delay
        )
        
        results['madqn_lp'] = {
            'total_delay_semantic': madqn_lp_delay,
            'k_v2v': madqn_k_v2v,
            'k_v2i': madqn_k_v2i,
            'lambda_local': madqn_lambda_local,
            'lambda_edge': madqn_lambda_edge,
            'lambda_bs': madqn_lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': madqn_v2v_sem_delay,
            'service_computation_delay': madqn_service_comp_delay,
            'v2i_transmission_delay': madqn_v2i_sem_delay,
            'bs_computation_delay': madqn_bs_comp_delay
        }
        
        # ================== 方案3: Ivy联合优化（注释掉） ==================
        # 使用Ivy算法同时优化k值和lambda值
        # try:
        #     ivy_k, ivy_lambda, ivy_delay = self.ivy_optimizer.quick_optimize(
        #         snr_v2v, snr_v2i, task_size, env, service_vehicle_tasks, total_tasks_in_slot
        #     )
        #     ivy_k_v2v, ivy_k_v2i = ivy_k
        #     ivy_lambda_local, ivy_lambda_edge, ivy_lambda_bs = ivy_lambda
        #     
        #     # 计算Ivy方案的延迟分解（固定任务大小，用于速率对比）
        #     ivy_local_delay = local_delay  # 本地延迟不变
        #     _, ivy_v2v_delay = env.calculate_semantic_rate_and_delay(snr_v2v, ivy_k_v2v, task_size)
        #     ivy_service_comp_delay = env.calculate_computation_delay(task_size, env.service_cpu_freq, service_vehicle_tasks)
        #     _, ivy_v2i_delay = env.calculate_semantic_rate_and_delay(snr_v2i, ivy_k_v2i, task_size)
        #     ivy_bs_comp_delay = env.calculate_computation_delay(task_size, env.bs_cpu_freq, total_tasks_in_slot)
        #     
        # except Exception as e:
        #     # 如果Ivy优化失败，使用MAPPO的结果作为备选
        #     print(f"⚠️ Ivy优化失败，使用MAPPO结果: {e}")
        #     ivy_k_v2v, ivy_k_v2i = mappo_k_v2v, mappo_k_v2i
        #     ivy_lambda_local, ivy_lambda_edge, ivy_lambda_bs = lambda_local, lambda_edge, lambda_bs
        #     ivy_delay = mappo_lp_delay
        #     ivy_local_delay = local_delay
        #     ivy_v2v_delay = v2v_sem_delay
        #     ivy_service_comp_delay = service_comp_delay
        #     ivy_v2i_delay = v2i_sem_delay
        #     ivy_bs_comp_delay = bs_comp_delay
        # 
        # results['ivy_joint'] = {
        #     'total_delay_semantic': ivy_delay,
        #     'k_v2v': ivy_k_v2v,
        #     'k_v2i': ivy_k_v2i,
        #     'lambda_local': ivy_lambda_local,
        #     'lambda_edge': ivy_lambda_edge,
        #     'lambda_bs': ivy_lambda_bs,
        #     'snr_v2v': snr_v2v,
        #     'snr_v2i': snr_v2i,
        #     # 延迟分解
        #     'local_delay': ivy_local_delay,
        #     'v2v_transmission_delay': ivy_v2v_delay,
        #     'service_computation_delay': ivy_service_comp_delay,
        #     'v2i_transmission_delay': ivy_v2i_delay,
        #     'bs_computation_delay': ivy_bs_comp_delay
        # }
        
        # ================== 方案4: 语义-Ivy ==================
        # 使用MAPPO k值，但用Ivy算法优化lambda分配
        
        # 重新计算MAPPO k值和各分支延迟（防止之前方案被注释导致变量未定义）
        mappo_k_v2v = self.get_mappo_k_value(snr_v2v, task_size, env.semantic_table, 'v2v', v2v_distance)
        mappo_k_v2i = self.get_mappo_k_value(snr_v2i, task_size, env.semantic_table, 'v2i', v2i_distance)
        
        # V2V + 服务车辆分支延迟
        _, v2v_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2v, mappo_k_v2v, task_size)
        edge_delay = v2v_sem_delay + service_comp_delay
        
        # V2I + 基站分支延迟  
        _, v2i_sem_delay = env.calculate_semantic_rate_and_delay(snr_v2i, mappo_k_v2i, task_size)
        bs_delay = v2i_sem_delay + bs_comp_delay
        
        # 检查输入参数的有效性
        if local_delay <= 0 or edge_delay <= 0 or bs_delay <= 0:
            raise ValueError(f"延迟参数无效: local_delay={local_delay}, edge_delay={edge_delay}, bs_delay={bs_delay}")
        
        # 使用Ivy算法优化lambda分配
        semantic_ivy_lambda_local, semantic_ivy_lambda_edge, semantic_ivy_lambda_bs, semantic_ivy_delay, ivy_info = \
            self.ivy_lambda_optimizer.optimize_lambda(local_delay, edge_delay, bs_delay, verbose=False)
        
        # 检查输出的有效性
        lambda_sum = semantic_ivy_lambda_local + semantic_ivy_lambda_edge + semantic_ivy_lambda_bs
        if abs(lambda_sum - 1.0) > 1e-6:
            print(f"[WARNING] Lambda分配不符合约束: sum={lambda_sum:.8f}")
        
        if semantic_ivy_delay <= 0:
            raise ValueError(f"Ivy优化结果无效: semantic_ivy_delay={semantic_ivy_delay}")
        
        results['semantic_ivy'] = {
            'total_delay_semantic': semantic_ivy_delay,
            'k_v2v': mappo_k_v2v,
            'k_v2i': mappo_k_v2i,
            'lambda_local': semantic_ivy_lambda_local,
            'lambda_edge': semantic_ivy_lambda_edge,
            'lambda_bs': semantic_ivy_lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_sem_delay,
            'service_computation_delay': service_comp_delay,
            'v2i_transmission_delay': v2i_sem_delay,
            'bs_computation_delay': bs_comp_delay
        }
        
        # ================== 方案5: 语义方案（无LP，等权分配） ==================
        # 使用相同的MAPPO k值，但等权分配lambda（不优化）
        
        # 计算等权分配下各分支的实际延迟
        local_actual_delay = local_delay * 1/2
        edge_actual_delay = edge_delay * 1/4 
        bs_actual_delay = bs_delay * 1/4
        
        # 最终延迟是最慢分支的时间（正确的并行计算逻辑）
        semantic_no_lp_delay = max(local_actual_delay, edge_actual_delay, bs_actual_delay)
        
        results['semantic_no_lp'] = {
            'total_delay_semantic': semantic_no_lp_delay,
            'k_v2v': mappo_k_v2v,
            'k_v2i': mappo_k_v2i,
            'lambda_local': 1/2,
            'lambda_edge': 1/4,
            'lambda_bs': 1/4,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_sem_delay,
            'service_computation_delay': service_comp_delay,
            'v2i_transmission_delay': v2i_sem_delay,
            'bs_computation_delay': bs_comp_delay
        }
        
        # ================== 方案5: 香农方案 ==================
        # 使用k=1（香农容量）
        shannon_k_v2v = 1
        shannon_k_v2i = 1
        
        # 重新计算各分支延迟
        shannon_local_delay = local_delay  # 本地计算不变
        
        # V2V + 服务车辆（香农）
        _, v2v_shannon_delay = env.calculate_shannon_rate_and_delay(snr_v2v, task_size)
        shannon_service_comp_delay = env.calculate_computation_delay(task_size, env.service_cpu_freq, service_vehicle_tasks)
        shannon_edge_delay = v2v_shannon_delay + shannon_service_comp_delay
        
        # V2I + 基站（香农）
        _, v2i_shannon_delay = env.calculate_shannon_rate_and_delay(snr_v2i, task_size)
        shannon_bs_comp_delay = env.calculate_computation_delay(task_size, env.bs_cpu_freq, total_tasks_in_slot)
        shannon_bs_delay = v2i_shannon_delay + shannon_bs_comp_delay
        
        # 使用LP求解香农方案的最优lambda
        shannon_lambda_local, shannon_lambda_edge, shannon_lambda_bs, shannon_optimal_delay = compute_optimal_lambda(
            shannon_local_delay, shannon_edge_delay, shannon_bs_delay
        )
        
        results['shannon'] = {
            'total_delay_shannon': shannon_optimal_delay,
            'k_v2v': shannon_k_v2v,
            'k_v2i': shannon_k_v2i,
            'lambda_local': shannon_lambda_local,
            'lambda_edge': shannon_lambda_edge,
            'lambda_bs': shannon_lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_shannon_delay,
            'service_computation_delay': shannon_service_comp_delay,
            'v2i_transmission_delay': v2i_shannon_delay,
            'bs_computation_delay': shannon_bs_comp_delay
        }
        
        # ================== 方案7: 香农-Ivy ==================
        # 使用香农k值（k=1），但用Ivy算法优化lambda分配
        
        # 检查香农方案的延迟参数
        if shannon_local_delay <= 0 or shannon_edge_delay <= 0 or shannon_bs_delay <= 0:
            raise ValueError(f"香农延迟参数无效: local={shannon_local_delay}, edge={shannon_edge_delay}, bs={shannon_bs_delay}")
        
        shannon_ivy_lambda_local, shannon_ivy_lambda_edge, shannon_ivy_lambda_bs, shannon_ivy_delay, shannon_ivy_info = \
            self.ivy_lambda_optimizer.optimize_lambda(shannon_local_delay, shannon_edge_delay, shannon_bs_delay, verbose=False)
        
        # 检查输出的有效性
        shannon_lambda_sum = shannon_ivy_lambda_local + shannon_ivy_lambda_edge + shannon_ivy_lambda_bs
        if abs(shannon_lambda_sum - 1.0) > 1e-6:
            print(f"[WARNING] 香农-Ivy Lambda分配不符合约束: sum={shannon_lambda_sum:.8f}")
        
        if shannon_ivy_delay <= 0:
            raise ValueError(f"香农-Ivy优化结果无效: shannon_ivy_delay={shannon_ivy_delay}")
        
        results['shannon_ivy'] = {
            'total_delay_shannon': shannon_ivy_delay,
            'k_v2v': shannon_k_v2v,
            'k_v2i': shannon_k_v2i,
            'lambda_local': shannon_ivy_lambda_local,
            'lambda_edge': shannon_ivy_lambda_edge,
            'lambda_bs': shannon_ivy_lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_shannon_delay,
            'service_computation_delay': shannon_service_comp_delay,
            'v2i_transmission_delay': v2i_shannon_delay,
            'bs_computation_delay': shannon_bs_comp_delay
        }
        
        # ================== 方案8: 香农方案（无LP，等权分配） ==================
        # 使用香农k值（k=1），但等权分配lambda（不优化）
        
        # 计算等权分配下各分支的实际延迟
        shannon_local_actual_delay = shannon_local_delay * 1/2
        shannon_edge_actual_delay = shannon_edge_delay * 1/4
        shannon_bs_actual_delay = shannon_bs_delay * 1/4
        
        # 异常值保护机制：当传输延迟超过3000ms时，设为3000ms
        OUTLIER_THRESHOLD = 3.0  # 3秒 = 3000ms
        
        # 检查并限制各分支延迟
        if shannon_local_actual_delay > OUTLIER_THRESHOLD:
            shannon_local_actual_delay = OUTLIER_THRESHOLD
            
        if shannon_edge_actual_delay > OUTLIER_THRESHOLD:
            shannon_edge_actual_delay = OUTLIER_THRESHOLD
            
        if shannon_bs_actual_delay > OUTLIER_THRESHOLD:
            shannon_bs_actual_delay = OUTLIER_THRESHOLD
        
        # 最终延迟是最慢分支的时间（正确的并行计算逻辑）
        shannon_no_lp_delay = max(shannon_local_actual_delay, shannon_edge_actual_delay, shannon_bs_actual_delay)
        
        results['shannon_no_lp'] = {
            'total_delay_shannon': shannon_no_lp_delay,
            'k_v2v': shannon_k_v2v,
            'k_v2i': shannon_k_v2i,
            'lambda_local': 1/2,
            'lambda_edge': 1/4,
            'lambda_bs': 1/4,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_shannon_delay,
            'service_computation_delay': shannon_service_comp_delay,
            'v2i_transmission_delay': v2i_shannon_delay,
            'bs_computation_delay': shannon_bs_comp_delay
        }
        
        # ================== 方案9: 传统k选择 + LP ==================
        traditional_k_v2v = self.get_traditional_k(snr_v2v)
        traditional_k_v2i = self.get_traditional_k(snr_v2i)
        
        # 重新计算各分支延迟
        traditional_local_delay = local_delay  # 本地计算不变
        
        # V2V + 服务车辆（传统k）
        _, v2v_trad_delay = env.calculate_semantic_rate_and_delay(snr_v2v, traditional_k_v2v, task_size)
        traditional_service_comp_delay = env.calculate_computation_delay(task_size, env.service_cpu_freq, service_vehicle_tasks)
        traditional_edge_delay = v2v_trad_delay + traditional_service_comp_delay
        
        # V2I + 基站（传统k）
        _, v2i_trad_delay = env.calculate_semantic_rate_and_delay(snr_v2i, traditional_k_v2i, task_size)
        traditional_bs_comp_delay = env.calculate_computation_delay(task_size, env.bs_cpu_freq, total_tasks_in_slot)
        traditional_bs_delay = v2i_trad_delay + traditional_bs_comp_delay
        
        # 使用LP求解传统k方案的最优lambda
        trad_lambda_local, trad_lambda_edge, trad_lambda_bs, traditional_optimal_delay = compute_optimal_lambda(
            traditional_local_delay, traditional_edge_delay, traditional_bs_delay
        )
        
        results['traditional_k'] = {
            'total_delay_semantic': traditional_optimal_delay,
            'k_v2v': traditional_k_v2v,
            'k_v2i': traditional_k_v2i,
            'lambda_local': trad_lambda_local,
            'lambda_edge': trad_lambda_edge,
            'lambda_bs': trad_lambda_bs,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            # 延迟分解（固定任务大小，用于速率对比）
            'local_delay': local_delay,
            'v2v_transmission_delay': v2v_trad_delay,
            'service_computation_delay': traditional_service_comp_delay,
            'v2i_transmission_delay': v2i_trad_delay,
            'bs_computation_delay': traditional_bs_comp_delay
        }
        
        return results
    
    def run_optimized_comparison(self, n_steps=100, debug_mode=True):
        """运行优化的对比测试 - 一次仿真，多种方案计算"""
        print(f"\n🔬 开始优化性能对比测试...")
        print(f"📊 一次运行环境({n_steps}步)，计算所有方案性能")
        if debug_mode:
            print("🔍 调试模式：将验证任务-服务车辆对应关系")
        print("=" * 60)
        
        try:
            # 创建环境 - 完全模仿HighwayEnvironment.py的test_performance_analysis
            print("🏗️ 初始化环境...")
            env = HighwayEnvironment(n_task_vehicles=self.n_task_vehicles, n_service_vehicles=self.n_service_vehicles, random_seed=42, task_size=self.task_size)
            env.enable_building_loss = False  # 禁用建筑物损耗
            env.enable_lp = True  # 启用LP（但我们会手动控制）
            print(f"✅ 环境初始化完成：{self.n_task_vehicles}任务车，{self.n_service_vehicles}服务车")
            
            # 运行仿真获取基础数据
            print(f"🏃 运行{n_steps}个时隙的仿真，收集SNR数据...")
            base_results = env.run_simulation(n_steps)
            
            if not base_results:
                print("⚠️ 仿真期间未生成任务")
                return
            
            print(f"✅ 基础仿真完成：处理{len(base_results)}个任务")
            
            # 基于仿真结果计算所有方案的性能
            print("🧮 基于相同SNR数据计算所有方案性能...")
            
            all_results = {
                'mappo_snr_lp': [],  # 新增MAPPO-SNR方案，放在首位
                'mappo_lp': [],
                'madqn_lp': [],
                'semantic_ivy': [],  # 语义-ivy方案
                'semantic_no_lp': [],
                'shannon': [],
                'shannon_ivy': [],  # 香农-ivy方案
                'shannon_no_lp': [],
                'traditional_k': []
            }
            
            total_tasks = len(base_results)
            for i, base_result in enumerate(base_results):
                if i % 10 == 0 or i == total_tasks - 1:  # 每10个任务打印一次进度
                    print(f"  处理进度: {i+1}/{total_tasks} ({(i+1)/total_tasks*100:.1f}%)")
                
                try:
                    # 提取基础数据
                    snr_v2v = base_result['snr_v2v']
                    snr_v2i = base_result['snr_v2i']
                    task_size = self.task_size  # 使用设定的固定任务大小
                    service_vehicle_tasks = base_result.get('service_vehicle_tasks', 1)
                    total_tasks_in_slot = base_result.get('total_tasks_in_slot', 1)
                    
                    # 调试模式：验证对应关系
                    if debug_mode and i < 5:  # 只打印前5个任务的信息
                        print(f"  任务{i+1}: 任务车辆{base_result.get('task_vehicle_id', 'N/A')} -> "
                              f"服务车辆{base_result.get('service_vehicle_id', 'N/A')} "
                              f"(该服务车辆任务数: {service_vehicle_tasks}, 时隙总任务数: {total_tasks_in_slot})")
                    
                    # 计算所有方案的延迟
                    scheme_results = self.calculate_delays_for_all_schemes(
                        snr_v2v, snr_v2i, task_size, env, 
                        service_vehicle_tasks, total_tasks_in_slot,
                        base_result.get('task_vehicle_id'), base_result.get('service_vehicle_id')
                    )
                    
                    # 存储结果
                    for scheme_name, scheme_result in scheme_results.items():
                        all_results[scheme_name].append(scheme_result)
                        
                except Exception as e:
                    print(f"⚠️ 处理任务{i+1}时出错:")
                    print(f"  错误类型: {type(e).__name__}")
                    print(f"  错误信息: {str(e)}")
                    print(f"  SNR: V2V={snr_v2v:.2f}dB, V2I={snr_v2i:.2f}dB")
                    print(f"  返回的scheme_results键: {list(scheme_results.keys()) if 'scheme_results' in locals() else 'N/A'}")
                    
                    import traceback
                    print(f"  详细错误堆栈:")
                    traceback.print_exc()
                    continue
            
            # 保存结果
            self.results = all_results
            
            print("✅ 所有方案计算完成！")
            self._print_summary()
            
        except Exception as e:
            print(f"❌ 运行过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_summary(self):
        """打印测试总结"""
        print(f"\n📊 优化测试总结 (车辆数: {self.n_task_vehicles}):")
        print("=" * 70)
        
        method_names = {
            'mappo_snr_lp': 'MAPPO-SNR + LP',  # 新增MAPPO-SNR方案，放在首位
            'mappo_lp': 'MAPPO + LP',
            'madqn_lp': 'MADQN + LP',
            'semantic_ivy': 'Semantic + Ivy',  # 新增语义-ivy方案
            'semantic_no_lp': 'Semantic (No LP)',
            'shannon': 'Shannon + LP',
            'shannon_ivy': 'Shannon + Ivy',  # 新增香农-ivy方案
            'shannon_no_lp': 'Shannon (No LP)',
            'traditional_k': 'Traditional K-Selection'
        }
        
        for method_name, results in self.results.items():
            if results:
                # 根据方案选择合适的时延字段
                if method_name in ['shannon', 'shannon_ivy', 'shannon_no_lp']:
                    # 添加异常值保护机制：过滤大于3000ms的香农传输时间
                    all_delays = np.array([r['total_delay_shannon'] for r in results]) * 1000
                    delays = all_delays[all_delays <= 3000]  # 过滤异常值
                    if len(delays) == 0:  # 如果全部都是异常值，使用原始数据避免程序崩溃
                        delays = all_delays
                else:
                    delays = np.array([r['total_delay_semantic'] for r in results]) * 1000
                
                display_name = method_names[method_name]
                
                print(f"{display_name:15s}: "
                      f"任务数={len(results):3d}, "
                      f"平均={delays.mean():.2f}ms, "
                      f"标准差={delays.std():.2f}ms, "
                      f"最小={delays.min():.2f}ms, "
                      f"最大={delays.max():.2f}ms")
                

        
        # 计算改进幅度
        if self.results['mappo_lp'] and self.results['shannon']:
            mappo_delays = [r['total_delay_semantic'] for r in self.results['mappo_lp']]
            # 添加异常值保护：过滤香农方案中大于3秒的延迟值
            all_shannon_delays = [r['total_delay_shannon'] for r in self.results['shannon']]
            shannon_delays = [d for d in all_shannon_delays if d * 1000 <= 3000]
            if len(shannon_delays) == 0:
                shannon_delays = all_shannon_delays  # 如果全部异常，使用原始数据
            
            mappo_avg = np.mean(mappo_delays) * 1000
            shannon_avg = np.mean(shannon_delays) * 1000
            improvement = (shannon_avg - mappo_avg) / shannon_avg * 100
            print(f"\n🎯 MAPPO+LP相比香农方案改进: {improvement:.2f}%")
        
        # 计算相比传统k选择的改进
        if self.results['mappo_lp'] and self.results['traditional_k']:
            mappo_delays = [r['total_delay_semantic'] for r in self.results['mappo_lp']]
            traditional_delays = [r['total_delay_semantic'] for r in self.results['traditional_k']]
            
            mappo_avg = np.mean(mappo_delays) * 1000
            traditional_avg = np.mean(traditional_delays) * 1000
            improvement = (traditional_avg - mappo_avg) / traditional_avg * 100
            print(f"🎯 MAPPO+LP相比传统k选择改进: {improvement:.2f}%")
        
        # 计算MADQN与MAPPO的对比
        if self.results['madqn_lp'] and self.results['mappo_lp']:
            madqn_delays = [r['total_delay_semantic'] for r in self.results['madqn_lp']]
            mappo_delays = [r['total_delay_semantic'] for r in self.results['mappo_lp']]
            
            madqn_avg = np.mean(madqn_delays) * 1000
            mappo_avg = np.mean(mappo_delays) * 1000
            
            if madqn_avg < mappo_avg:
                improvement = (mappo_avg - madqn_avg) / mappo_avg * 100
                print(f"🎯 MADQN+LP相比MAPPO+LP改进: {improvement:.2f}%")
            else:
                decline = (madqn_avg - mappo_avg) / mappo_avg * 100
                print(f"📊 MADQN+LP相比MAPPO+LP差距: {decline:.2f}%")
        
        # 计算Ivy联合优化与MAPPO+LP的对比（注释掉）
        # if self.results['ivy_joint'] and self.results['mappo_lp']:
        #     ivy_delays = [r['total_delay_semantic'] for r in self.results['ivy_joint']]
        #     mappo_delays = [r['total_delay_semantic'] for r in self.results['mappo_lp']]
        #     
        #     ivy_avg = np.mean(ivy_delays) * 1000
        #     mappo_avg = np.mean(mappo_delays) * 1000
        #     
        #     if ivy_avg < mappo_avg:
        #         improvement = (mappo_avg - ivy_avg) / mappo_avg * 100
        #         print(f"🌟 Ivy联合优化相比MAPPO+LP改进: {improvement:.2f}%")
        #     else:
        #         decline = (ivy_avg - mappo_avg) / mappo_avg * 100
        #         print(f"📊 Ivy联合优化相比MAPPO+LP差距: {decline:.2f}%")
        # 
        # # 计算Ivy联合优化与传统k选择的对比
        # if self.results['ivy_joint'] and self.results['traditional_k']:
        #     ivy_delays = [r['total_delay_semantic'] for r in self.results['ivy_joint']]
        #     traditional_delays = [r['total_delay_semantic'] for r in self.results['traditional_k']]
        #     
        #     ivy_avg = np.mean(ivy_delays) * 1000
        #     traditional_avg = np.mean(traditional_delays) * 1000
        #     improvement = (traditional_avg - ivy_avg) / traditional_avg * 100
        #     print(f"🌟 Ivy联合优化相比传统k选择改进: {improvement:.2f}%")
        
        # MADQN相比传统方案的改进
        if self.results['madqn_lp'] and self.results['traditional_k']:
            madqn_delays = [r['total_delay_semantic'] for r in self.results['madqn_lp']]
            traditional_delays = [r['total_delay_semantic'] for r in self.results['traditional_k']]
            
            madqn_avg = np.mean(madqn_delays) * 1000
            traditional_avg = np.mean(traditional_delays) * 1000
            improvement = (traditional_avg - madqn_avg) / traditional_avg * 100
            print(f"🎯 MADQN+LP相比传统k选择改进: {improvement:.2f}%")
        
        # 计算LP的作用
        if self.results['mappo_lp'] and self.results['semantic_no_lp']:
            mappo_lp_delays = [r['total_delay_semantic'] for r in self.results['mappo_lp']]
            no_lp_delays = [r['total_delay_semantic'] for r in self.results['semantic_no_lp']]
            
            mappo_lp_avg = np.mean(mappo_lp_delays) * 1000
            no_lp_avg = np.mean(no_lp_delays) * 1000
            lp_improvement = (no_lp_avg - mappo_lp_avg) / no_lp_avg * 100
            print(f"🎯 LP优化带来的改进: {lp_improvement:.2f}%")
    
    def save_results(self):
        """保存测试结果"""
        filename = f"optimized_comparison_{self.n_task_vehicles}vehicles.pkl"
        
        # 生成统计量，便于后续绘图显示极值/方差等
        stats = {}
        for method, records in self.results.items():
            if not records:
                continue
            if method in ['shannon', 'shannon_ivy', 'shannon_no_lp']:
                all_delays = np.array([r['total_delay_shannon'] for r in records]) * 1000
                delays = all_delays[all_delays <= 3000]
                if len(delays) == 0:
                    delays = all_delays
            else:
                delays = np.array([r['total_delay_semantic'] for r in records]) * 1000
            
            stats[method] = {
                'count': int(len(delays)),
                'mean': float(delays.mean()),
                'std': float(delays.std()),
                'min': float(delays.min()),
                'max': float(delays.max())
            }
        
        save_data = {
            'results': self.results,
            'n_task_vehicles': self.n_task_vehicles,
            'n_service_vehicles': self.n_service_vehicles,
            'test_type': 'optimized_single_simulation',
            'stats': stats
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 结果已保存到: {filename}")
        return filename
    
    def plot_comparison(self, save_path=None):
        """绘制对比图 - 堆叠柱状图显示传输延迟和计算延迟"""
        # Set font to support English labels
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        method_names = {
            'mappo_snr_lp': 'MAPPO-SNR + LP',  # 新增MAPPO-SNR方案，放在首位
            'mappo_lp': 'MAPPO + LP',
            'madqn_lp': 'MADQN + LP',
            'semantic_ivy': 'Semantic + Ivy',  # 新增语义-ivy方案
            'semantic_no_lp': 'Semantic (No LP)', 
            'shannon': 'Shannon + LP',
            'shannon_ivy': 'Shannon + Ivy',  # 新增香农-ivy方案
            'shannon_no_lp': 'Shannon (No LP)',
            'traditional_k': 'Traditional K-Selection'
        }
        
        methods = []
        transmission_delays = []
        computation_delays = []
        
        for method, display_name in method_names.items():
            if self.results[method]:
                methods.append(display_name)
                
                # 计算平均传输延迟和计算延迟
                trans_delays = []
                comp_delays = []
                
                for result in self.results[method]:
                    # 获取lambda权重
                    lambda_local = result['lambda_local']
                    lambda_edge = result['lambda_edge'] 
                    lambda_bs = result['lambda_bs']
                    
                    # 计算各分支的完整延迟
                    local_total = result['local_delay'] * lambda_local
                    edge_total = (result['v2v_transmission_delay'] + result['service_computation_delay']) * lambda_edge
                    bs_total = (result['v2i_transmission_delay'] + result['bs_computation_delay']) * lambda_bs
                    
                    # 找到最大延迟的分支
                    max_delay = max(local_total, edge_total, bs_total)
                    
                    # 只统计最大延迟分支的传输和计算延迟
                    if abs(local_total - max_delay) < 1e-9:  # 本地计算最大
                        trans_delay = 0  # 无传输延迟
                        comp_delay = result['local_delay'] * lambda_local
                    elif abs(edge_total - max_delay) < 1e-9:  # 边缘计算最大
                        trans_delay = result['v2v_transmission_delay'] * lambda_edge
                        comp_delay = result['service_computation_delay'] * lambda_edge
                    else:  # 基站计算最大
                        trans_delay = result['v2i_transmission_delay'] * lambda_bs
                        comp_delay = result['bs_computation_delay'] * lambda_bs
                    
                    trans_delays.append(trans_delay * 1000)  # 转换为毫秒
                    comp_delays.append(comp_delay * 1000)    # 转换为毫秒
                
                transmission_delays.append(np.mean(trans_delays))
                computation_delays.append(np.mean(comp_delays))
        
        if methods:
            x = np.arange(len(methods))
            width = 0.6
            
            # 为不同方法定义不同图案，便于灰白色打印区分
            hatches = ['///', '\\\\\\', '...', 'xxx', '---', '|||', '+++', 'ooo', '***']
            colors_comp = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#E74C3C', '#1ABC9C', '#34495E', '#E67E22', '#95A5A6']
            colors_trans = ['#5DADE2', '#58D68D', '#BB8FCE', '#F8C471', '#EC7063', '#48C9B0', '#5D6D7E', '#F0B27A', '#AEB6BF']
            
            # 绘制堆叠柱状图，为每个方法使用不同图案（去除数字标识）
            for i in range(len(methods)):
                hatch_pattern = hatches[i % len(hatches)]
                
                # 计算延迟柱状图
                ax.bar(x[i], computation_delays[i], width, 
                      color=colors_comp[i % len(colors_comp)], 
                      alpha=0.8, hatch=hatch_pattern, 
                      edgecolor='black', linewidth=1.2)
                ax.bar(x[i], transmission_delays[i], width, 
                      bottom=computation_delays[i],
                      color=colors_trans[i % len(colors_trans)], 
                      alpha=0.8, hatch=hatch_pattern,
                      edgecolor='black', linewidth=1.2)
            
            # 添加图例（使用第一个条形的样式）
            ax.bar([], [], color=colors_comp[0], alpha=0.8, hatch=hatches[0], 
                  edgecolor='black', linewidth=1.2, label='Computation Delay')
            ax.bar([], [], color=colors_trans[0], alpha=0.8, hatch=hatches[0],
                  edgecolor='black', linewidth=1.2, label='Transmission Delay')
            
            ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Task Delay (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'Average Task Delay Comparison ({self.n_task_vehicles} vehicles)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha='right')
            ax.legend(loc='upper left', frameon=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(np.array(computation_delays) + np.array(transmission_delays)) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_transmission_delay_comparison(self, save_path=None):
        """绘制不同方案各部分传输时延对比图"""
        # Set font to support English labels
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 包含新增的Ivy方案
        method_names = {
            'mappo_snr_lp': 'MAPPO-SNR + LP',  # 新增MAPPO-SNR方案，放在首位
            'mappo_lp': 'MAPPO + LP',
            'madqn_lp': 'MADQN + LP',
            'semantic_ivy': 'Semantic + Ivy',  # 新增语义-ivy方案
            'semantic_no_lp': 'Semantic (No LP)',
            'shannon': 'Shannon + LP',
            'shannon_ivy': 'Shannon + Ivy',  # 新增香农-ivy方案
            'shannon_no_lp': 'Shannon (No LP)',
            'traditional_k': 'Traditional K-Selection'
        }
        
        # 传输类型的固定颜色（区分度高的颜色）
        transmission_colors = {
            'v2v': '#E74C3C',    # 红色 - V2V传输
            'v2i': '#3498DB',    # 蓝色 - V2I传输
            'local': '#95A5A6'   # 灰色 - 本地（无传输）
        }
        
        methods = []
        v2v_delays = []
        v2i_delays = []
        
        for method, display_name in method_names.items():
            if self.results[method]:
                methods.append(display_name)
                
                # 计算各部分的平均传输延迟
                v2v_trans = []
                v2i_trans = []
                
                for result in self.results[method]:
                    # 获取lambda权重
                    lambda_edge = result['lambda_edge'] 
                    lambda_bs = result['lambda_bs']
                    
                    # V2V传输延迟（加权）
                    v2v_delay = result['v2v_transmission_delay'] * lambda_edge
                    v2v_delay_ms = v2v_delay * 1000  # 转换为毫秒
                    
                    # V2I传输延迟（加权）
                    v2i_delay = result['v2i_transmission_delay'] * lambda_bs
                    v2i_delay_ms = v2i_delay * 1000  # 转换为毫秒
                    
                    # 添加异常值保护：对香农方案的传输延迟进行过滤
                    if method in ['shannon', 'shannon_no_lp', 'shannon_ivy']:
                        # 过滤掉大于3000ms的传输延迟
                        if v2v_delay_ms <= 3000 and v2i_delay_ms <= 3000:
                            v2v_trans.append(v2v_delay_ms)
                            v2i_trans.append(v2i_delay_ms)
                    else:
                        v2v_trans.append(v2v_delay_ms)
                        v2i_trans.append(v2i_delay_ms)
                
                v2v_delays.append(np.mean(v2v_trans))
                v2i_delays.append(np.mean(v2i_trans))
        
        if methods:
            x = np.arange(len(methods))
            
            # 计算总传输延迟（V2V + V2I）
            total_delays = np.array(v2v_delays) + np.array(v2i_delays)
            
            # 为每种方法定义不同的标记和颜色，增强视觉区分度
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', 'X']
            colors = ['#FFD700', '#2E86C1', '#1ABC9C', '#F39C12', '#E74C3C', '#9B59B6', '#2ECC71', '#34495E', '#E67E22', '#FF69B4', '#8A2BE2']
            
            # 绘制连接线（淡化背景）
            ax.plot(x, total_delays, color='lightgray', alpha=0.4, linewidth=1.5, zorder=1)
            
            # 绘制各个方法的数据点，去除数字标识
            for i, (method, total_delay) in enumerate(zip(methods, total_delays)):
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                # 使用更大的标记点和更明显的边框
                ax.scatter(i, total_delay, marker=marker, color=color, 
                          s=150, linewidth=2.8, label=method,
                          edgecolors='black', zorder=3, alpha=0.9)
            
            ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Transmission Delay (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'Total Transmission Delay Comparison ({self.n_task_vehicles} vehicles)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
            
            # 优化图例位置和样式
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                     frameon=True, framealpha=0.9, edgecolor='black')
            
            # 添加网格线提高可读性
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim(0, max(total_delays) * 1.15)  # 留出更多顶部空间
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 传输延迟对比图已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_multi_vehicle_comparison(all_results, save_path=None):
        """绘制多车辆数的综合对比图"""
        # Set font to support English labels
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        def _filter_outliers_iqr(arr, whisker=1.5):
            """IQR过滤：统计意义的异常值剔除，若全剔除则回退原数组"""
            if len(arr) == 0:
                return arr
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - whisker * iqr
            upper = q3 + whisker * iqr
            filtered = arr[(arr >= lower) & (arr <= upper)]
            return filtered if len(filtered) > 0 else arr
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        
        # 提取数据
        vehicle_counts = sorted(all_results.keys())
        method_names = {
            # LP组：先语义后香农
            'mappo_snr_lp': 'Our-Method + LP',  # 语义LP
            'shannon': 'Traditional + LP',          # 香农LP
            
            # Ivy组：先语义后香农
            'semantic_ivy': 'MAPPO + Ivy',   # 语义Ivy
            'shannon_ivy': 'Traditional + Ivy',     # 香农Ivy
            
            # NoLP组：先语义后香农
            'semantic_no_lp': 'MAPPO (No LP)',  # 语义NoLP
            'shannon_no_lp': 'Traditional (No LP)',     # 香农NoLP
            
            #'mappo_lp': 'MAPPO + LP',
            #'madqn_lp': 'MADQN + LP',
            #'traditional_k': 'Traditional K-Selection'
        }
        
        # 使用与第二个图一致的配色方案，MAPPO-SNR使用显著的金色
        colors = {
            'mappo_snr_lp': '#FFD700',  # 金色 - MAPPO-SNR专用，高显著性
            #'mappo_lp': '#2E86C1',      # 蓝色系 - 与第二个图一致
            #'madqn_lp': '#1ABC9C',      # 青色系 - 与第二个图一致
            'semantic_ivy': '#FF1493',  # 深粉红色 - 语义-Ivy算法专用，高区分度
            'semantic_no_lp': '#8E44AD', # 紫色 - 第一个图独有的方法，区分度高
            'shannon': '#F39C12',       # 橙色系 - 与第二个图一致
            'shannon_ivy': '#FF6347',   # 番茄红色 - 香农-Ivy算法专用
            'shannon_no_lp': '#FF6B35', # 橙红色系 - 与第二个图一致
            #'traditional_k': '#E74C3C' # 红色系 - 与第二个图一致
        }
        
        # 计算柱状图位置
        n_methods = len(method_names)
        n_vehicles = len(vehicle_counts)
        bar_width = 0.12
        x_base = np.arange(n_vehicles)
        
        # 为不同方法定义不同图案，便于灰白色打印区分
        # 选用更简洁的图案，去掉实心星形，改用中空圆点样式
        hatches = ['///', '\\\\\\', '...', 'xxx', '--', '+']
        
        # 为每种方法收集数据并绘制柱状图
        for i, (method, display_name) in enumerate(method_names.items()):
            avg_delays = []
            min_delays = []
            max_delays = []
            std_delays = []
            for n_vehicles in vehicle_counts:
                if method in all_results[n_vehicles] and all_results[n_vehicles][method]:
                    if method in ['shannon', 'shannon_no_lp', 'shannon_ivy']:
                        # 添加异常值保护：过滤大于3000ms的香农传输时间
                        all_delays = [r['total_delay_shannon'] * 1000 for r in all_results[n_vehicles][method]]
                        delays = [d for d in all_delays if d <= 3000]
                        if len(delays) == 0:  # 如果全部异常，使用原始数据
                            delays = all_delays
                    else:
                        delays = [r['total_delay_semantic'] * 1000 for r in all_results[n_vehicles][method]]
                    # 在已有3000ms过滤基础上，再做IQR过滤以得到统计意义上的分布
                    delays_filtered = _filter_outliers_iqr(np.array(delays))
                    avg_delays.append(np.mean(delays_filtered))
                    std_delays.append(np.std(delays_filtered))
                    min_delays.append(np.min(delays_filtered))
                    max_delays.append(np.max(delays_filtered))
                else:
                    avg_delays.append(0)
                    std_delays.append(0)
                    min_delays.append(np.nan)
                    max_delays.append(np.nan)
            
            # 计算柱子位置
            x_pos = x_base + (i - (n_methods-1)/2) * bar_width
            
            # 获取图案
            hatch_pattern = hatches[i % len(hatches)]
            
            # 绘制柱状图，添加图案，去除数字标识
            bars = ax1.bar(x_pos, avg_delays, bar_width, 
                          color=colors[method], alpha=0.8, label=display_name,
                          hatch=hatch_pattern, edgecolor='black', linewidth=1.0,
                          yerr=std_delays, capsize=4, ecolor='black', error_kw={'elinewidth':1.2})
        
        ax1.set_xlabel('Number of Vehicle Users', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Average Task Delay (ms)', fontsize=18, fontweight='bold')
        ax1.set_title('Average Task Delay Comparison by Vehicle Configuration', fontsize=18, fontweight='bold')
        ax1.set_xticks(x_base)
        ax1.set_xticklabels(vehicle_counts, fontsize=11)
        ax1.legend(loc='upper left', fontsize=16)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 删除子图2（相对性能提升）
        

        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 综合对比图已保存到: {save_path}")
        
        plt.show()

    @staticmethod
    def plot_multi_vehicle_transmission_comparison(all_results, save_path=None):
        """绘制多车辆数的传输延迟对比图（双子图：香农方案和其他方案分开）"""
        # Set font to support English labels
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取数据
        vehicle_counts = sorted(all_results.keys())
        
        # 分组方案名称
        shannon_methods = {
            'shannon': 'Traditional + LP',
            'shannon_ivy': 'Traditional + Ivy',
            'shannon_no_lp': 'Traditional (No LP)',
        }
        
        other_methods = {
            'mappo_snr_lp': 'Our-Method + LP',
            'mappo_lp': 'MAPPO + LP',
            'madqn_lp': 'MADQN + LP',
            'semantic_ivy': 'Our-Method + Ivy',
            'traditional_k': 'Linear K-Selection'
        }
        
        # 高级配色方案 - 改进颜色区分度
        method_colors = {
            # 香农方案 - 使用不同色系提高区分度
            'shannon': '#FF4500',       # 橙红色
            'shannon_ivy': '#DC143C',   # 深红色  
            'shannon_no_lp': '#B22222', # 火砖红色
            
            # 其他方案 - 使用区分度高的颜色
            'mappo_snr_lp': '#FFD700',  # 金色
            'mappo_lp': '#4169E1',      # 皇家蓝
            'madqn_lp': '#00CED1',      # 深青色
            'semantic_ivy': '#FF1493',  # 深粉红色
            'traditional_k': '#8B4513'  # 马鞍棕色
        }
        
        # 为每种方法定义不同的标记，增强视觉区分度
        method_markers = {
            # 香农方案 - 使用三角形系列
            'shannon': '^',         # 上三角
            'shannon_ivy': 'v',     # 下三角
            'shannon_no_lp': '<',   # 左三角
            
            # 其他方案 - 使用不同形状
            'mappo_snr_lp': 'o',    # 圆形
            'mappo_lp': 's',        # 方形
            'madqn_lp': 'D',        # 菱形
            'semantic_ivy': 'p',    # 五角形
            'traditional_k': '*'    # 星形
        }
        
        # 创建双子图 - 上下分布
        fig, (ax_shannon, ax_other) = plt.subplots(2, 1, figsize=(12, 8), 
                                                  gridspec_kw={'height_ratios': [1, 1]})
        
        # X轴位置
        x_positions = np.arange(len(vehicle_counts))
        
        # 收集数据的函数
        def collect_method_data(method_dict):
            data = {}
            for method in method_dict.keys():
                total_delays = []
                
                for n_vehicles_count in vehicle_counts:
                    if method in all_results[n_vehicles_count] and all_results[n_vehicles_count][method]:
                        results = all_results[n_vehicles_count][method]
                        
                        # 计算平均总传输延迟（V2V + V2I）
                        total_transmission_delays = []
                        
                        for result in results:
                            lambda_edge = result.get('lambda_edge', 0)
                            lambda_bs = result.get('lambda_bs', 0)
                            
                            v2v_trans = result.get('v2v_transmission_delay', 0) * 1000  # 转换为毫秒
                            v2i_trans = result.get('v2i_transmission_delay', 0) * 1000
                            
                            # 计算总传输延迟
                            total_trans = (v2v_trans * lambda_edge) + (v2i_trans * lambda_bs)
                            
                            # 添加异常值保护：对香农方案的传输延迟进行过滤
                            if method in ['shannon', 'shannon_no_lp', 'shannon_ivy']:
                                # 过滤掉大于3000ms的传输延迟
                                if total_trans <= 3000:
                                    total_transmission_delays.append(total_trans)
                            else:
                                total_transmission_delays.append(total_trans)
                        
                        if total_transmission_delays:
                            avg_total = np.mean(total_transmission_delays)
                        else:
                            avg_total = 0
                        
                        total_delays.append(avg_total)
                    else:
                        total_delays.append(0)
                
                data[method] = total_delays
            return data
        
        # 收集香农方案和其他方案的数据
        shannon_data = collect_method_data(shannon_methods)
        other_data = collect_method_data(other_methods)
        
        # 绘制香农方案子图（上子图）
        for method, display_name in shannon_methods.items():
            if method in shannon_data:
                total_delays = shannon_data[method]
                marker = method_markers[method]
                color = method_colors[method]
                
                # 绘制折线
                ax_shannon.plot(x_positions, total_delays, color=color, linewidth=2.0, 
                               marker=marker, markersize=8, markeredgecolor='black', 
                               markeredgewidth=1.5, label=display_name, alpha=0.9, zorder=3)
        
        # 绘制其他方案子图（下子图）
        for method, display_name in other_methods.items():
            if method in other_data:
                total_delays = other_data[method]
                marker = method_markers[method]
                color = method_colors[method]
                
                # 绘制折线
                ax_other.plot(x_positions, total_delays, color=color, linewidth=2.0, 
                             marker=marker, markersize=8, markeredgecolor='black', 
                             markeredgewidth=1.5, label=display_name, alpha=0.9, zorder=3)
        
        # 设置香农方案子图（上子图）
        ax_shannon.set_xticks(x_positions)
        ax_shannon.set_xticklabels([])  # 不显示X轴标签
        ax_shannon.tick_params(axis='y', labelsize=11)
        
        # 设置其他方案子图（下子图）
        ax_other.set_xlabel('Number of Vehicle Users', fontsize=18, fontweight='bold')
        ax_other.set_xticks(x_positions)
        ax_other.set_xticklabels(vehicle_counts, fontsize=11)
        ax_other.tick_params(axis='y', labelsize=11)
        
        # 设置总标题和Y轴标签（横跨两个子图）
        fig.suptitle('Total Transmission Delay Comparison by Vehicle Configuration', fontsize=18, fontweight='bold', y=0.95)
        fig.text(0.04, 0.5, 'Total Transmission Delay (ms)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
        
        # 自适应Y轴范围（根据数据决定起始点）
        shannon_delays = []
        for data in shannon_data.values():
            shannon_delays.extend(data)
        if shannon_delays:
            min_shannon = min(shannon_delays)
            max_shannon = max(shannon_delays)
            range_shannon = max_shannon - min_shannon
            ax_shannon.set_ylim(bottom=min_shannon - range_shannon * 0.1, 
                               top=max_shannon + range_shannon * 0.1)
        
        other_delays = []
        for data in other_data.values():
            other_delays.extend(data)
        if other_delays:
            min_other = min(other_delays)
            max_other = max(other_delays)
            range_other = max_other - min_other
            ax_other.set_ylim(bottom=min_other - range_other * 0.1, 
                             top=max_other + range_other * 0.1)
        
        # 分别在各自子图中设置图例
        # 香农方案子图的图例
        shannon_handles, shannon_labels = ax_shannon.get_legend_handles_labels()
        ax_shannon.legend(shannon_handles, shannon_labels, loc='upper left', fontsize=16, 
                         frameon=True, framealpha=0.9, edgecolor='black')
        
        # 其他方案子图的图例
        other_handles, other_labels = ax_other.get_legend_handles_labels()
        ax_other.legend(other_handles, other_labels, loc='upper left', fontsize=16, 
                       frameon=True, framealpha=0.9, edgecolor='black', ncol=2)
        
        # 添加网格线提高可读性
        ax_shannon.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax_other.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 调整布局以适应总标题和Y轴标签
        plt.tight_layout()
        plt.subplots_adjust(left=0.12, top=0.90)  # 为Y轴标签和总标题留出空间
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 多车辆传输延迟对比图已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_data_distribution_table(all_results, save_path=None):
        """绘制数据量分布表格"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置字体
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取数据
        vehicle_counts = sorted(all_results.keys())
        method_names = {
            'mappo_snr_lp': 'MAPPO-SNR + LP',  # 新增MAPPO-SNR方案，放在首位
            'mappo_lp': 'MAPPO + LP',
            'madqn_lp': 'MADQN + LP',
            'semantic_ivy': 'Semantic + Ivy',  # 新增语义-ivy方案
            'semantic_no_lp': 'Semantic Scheme (No LP)',
            'shannon': 'Shannon Scheme + LP',
            'shannon_ivy': 'Shannon + Ivy',  # 新增香农-ivy方案
            'shannon_no_lp': 'Shannon Scheme (No LP)',
            'traditional_k': 'Traditional K-Selection'
        }
        
        # 计算表格数据
        table_data = []
        row_labels = []
        
        for method, display_name in method_names.items():
            row_data = []
            for n_vehicles in vehicle_counts:
                if method in all_results[n_vehicles] and all_results[n_vehicles][method]:
                    results = all_results[n_vehicles][method]
                    
                    # 计算平均lambda值
                    avg_lambda_local = np.mean([r['lambda_local'] for r in results])
                    avg_lambda_edge = np.mean([r['lambda_edge'] for r in results])
                    avg_lambda_bs = np.mean([r['lambda_bs'] for r in results])
                    
                    # 计算数据量 (lambda * 0.8)
                    local_data = avg_lambda_local * 0.8
                    bs_data = avg_lambda_bs * 0.8
                    edge_data = avg_lambda_edge * 0.8
                    
                    # 格式化为字符串 (保留3位小数)
                    cell_content = f"{local_data:.3f}\n{bs_data:.3f}\n{edge_data:.3f}"
                    row_data.append(cell_content)
                else:
                    row_data.append("N/A")
            
            table_data.append(row_data)
            row_labels.append(display_name)
        
        # 创建表格
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 列标题
        col_labels = [f'{n} vehicles' for n in vehicle_counts]
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15] * len(vehicle_counts))
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.5)
        
        # 设置表头样式
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行标签样式
        for i in range(len(row_labels)):
            table[(i+1, -1)].set_facecolor('#2196F3')
            table[(i+1, -1)].set_text_props(weight='bold', color='white')
        
        # 设置数据单元格样式
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if (i + j) % 2 == 0:
                    table[(i+1, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i+1, j)].set_facecolor('#FFFFFF')
        
        # 添加标题
        plt.title('Data Volume Distribution (Mbit)\nLocal / Base Station / Edge Vehicle', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 添加说明文字
        plt.figtext(0.5, 0.02, 
                   'Each cell shows: Local Computing / Offloading to BS / Offloading to Edge Vehicle\n'
                   'Values calculated as: λ_j × 0.8 Mbit (average across all tasks)',
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 数据量分布表格已保存到: {save_path}")
        
        plt.show()


def test_single_config():
    """测试单个配置（调试用）"""
    # 设置固定随机种子确保结果可重现
    import numpy as np
    import random
    import torch
    
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    print(f"🔧 已设置固定随机种子: {RANDOM_SEED}")
    print("🔧 单配置测试模式")
    
    # 只测试一个小配置
    n_task, n_service = 10, 3
    
    print(f"🚗 测试配置: {n_task}辆任务车, {n_service}辆服务车")
    
    try:
        # 初始化对比器
        comparator = OptimizedPerformanceComparator(
            n_task_vehicles=n_task,
            n_service_vehicles=n_service,
            mappo_model_path="k_value_mappo_final.pth",
            mappo_snr_model_path="mappo-snr/k_value_mappo_snr_final.pth",
            madqn_model_path=f"k_value_madqn_final_20vehicles_per_exponential.pth"
        )
        
        # 运行优化对比测试（较少步数，开启调试模式）
        comparator.run_optimized_comparison(n_steps=50, debug_mode=True)  # 减少到50步，开启调试
        
        # 保存结果
        result_file = comparator.save_results()
        
        # 不在单配置测试中绘制图片，只在main函数结束时绘制最终图
        
        print(f"✅ 单配置测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 单配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 设置固定随机种子确保结果可重现
    import numpy as np
    import random
    import torch
    
    RANDOM_SEED = 44
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    print(f"🔧 已设置固定随机种子: {RANDOM_SEED}")
    print("🚀 开始优化高速公路环境性能对比测试")
    
    # 测试不同车辆数的场景
    vehicle_configs = [
        (20, 5),   # 25辆任务车，5辆服务车  
        (25, 5),   # 30辆任务车，5辆服务车  
        (30, 5),   # 35辆任务车，5辆服务车
        (35, 5),   # 40辆任务车，5辆服务车
        (40, 5)   # 45辆任务车，5辆服务车
    ]
    
    print(f"📋 计划测试 {len(vehicle_configs)} 种车辆配置: {vehicle_configs}")
    
    # 存储所有结果用于综合对比
    all_results = {}
    all_result_files = []
    
    for n_task, n_service in vehicle_configs:
        print(f"\n{'='*80}")
        print(f"🚗 测试配置: {n_task}辆任务车, {n_service}辆服务车")
        print(f"{'='*80}")
        
        # 初始化对比器
        comparator = OptimizedPerformanceComparator(
            n_task_vehicles=n_task,
            n_service_vehicles=n_service,
            mappo_model_path="k_value_mappo_final.pth",
            mappo_snr_model_path="mappo-snr/k_value_mappo_snr_final.pth",
            madqn_model_path=f"k_value_madqn_final_20vehicles_per_exponential.pth"
        )
        
        # 运行优化对比测试
        comparator.run_optimized_comparison(n_steps=100)  # 100个仿真步骤
        
        # 保存结果
        result_file = comparator.save_results()
        all_result_files.append(result_file)
        
        # 存储结果用于综合对比
        all_results[n_task] = comparator.results
        
        # 绘制单个配置的对比图
       # plot_path = f"single_comparison_plot_{n_task}vehicles.png"
       # transmission_plot_path = f"transmission_comparison_plot_{n_task}vehicles.png"
        
        #comparator.plot_comparison(save_path=plot_path)
        #comparator.plot_transmission_delay_comparison(save_path=transmission_plot_path)
        
        print(f"\n✅ {n_task}辆车的优化测试完成！")
    
    # 绘制综合对比图
    print(f"\n{'='*80}")
    print("📈 生成综合对比图...")
    print(f"{'='*80}")
    
    multi_plot_path = f"multi_vehicle_comparison.png"
    OptimizedPerformanceComparator.plot_multi_vehicle_comparison(all_results, save_path=multi_plot_path)
    
    # 生成多车辆传输延迟对比图
    transmission_plot_path = f"multi_vehicle_transmission_comparison.png"
    OptimizedPerformanceComparator.plot_multi_vehicle_transmission_comparison(all_results, save_path=transmission_plot_path)
    
    # 生成数据量分布表格
    table_plot_path = f"data_distribution_table.png"
    OptimizedPerformanceComparator.plot_data_distribution_table(all_results, save_path=table_plot_path)
    
    # 保存综合结果
    comprehensive_filename = f"comprehensive_results.pkl"
    comprehensive_data = {
        'all_results': all_results,
        'vehicle_configs': vehicle_configs,
        'result_files': all_result_files,
        'test_type': 'comprehensive_multi_vehicle'
    }
    
    with open(comprehensive_filename, 'wb') as f:
        pickle.dump(comprehensive_data, f)
    
    print(f"💾 综合结果已保存到: {comprehensive_filename}")
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("🎉 所有测试完成！综合总结:")
    print(f"{'='*80}")
    
    for n_task in sorted(all_results.keys()):
        results = all_results[n_task]
        if results['mappo_lp'] and results['shannon']:
            mappo_delays = [r['total_delay_semantic'] for r in results['mappo_lp']]
            # 添加异常值保护：过滤香农方案中大于3000ms的延迟值
            all_shannon_delays = [r['total_delay_shannon'] for r in results['shannon']]
            shannon_delays = [d for d in all_shannon_delays if d * 1000 <= 3000]
            if len(shannon_delays) == 0:
                shannon_delays = all_shannon_delays  # 如果全部异常，使用原始数据
            
            improvement = (np.mean(shannon_delays) - np.mean(mappo_delays)) / np.mean(shannon_delays) * 100
            
            print(f"📊 {n_task}辆任务车: MAPPO+LP相比香农方案改进 {improvement:.2f}%")
            print(f"   - MAPPO+LP平均延迟: {np.mean(mappo_delays)*1000:.2f}ms")
            print(f"   - 香农方案平均延迟: {np.mean(shannon_delays)*1000:.2f}ms")
            print(f"   - 处理任务总数: {len(results['mappo_lp'])}")
    
    print(f"\n🎯 测试文件生成:")
    print(f"   - 综合对比图: {multi_plot_path}")
    print(f"   - 传输延迟对比图: {transmission_plot_path}")
    print(f"   - 数据量分布表格: {table_plot_path}")
    print(f"   - 综合结果数据: {comprehensive_filename}")
    for i, (n_task, n_service) in enumerate(vehicle_configs):
        print(f"   - {n_task}辆车详细结果: {all_result_files[i]}")


def collect_data_only():
    """仅收集数据并保存，不绘图"""
    print("🔄 开始数据收集模式...")
    
    # 车辆配置列表
    vehicle_configs = [
        (10, 10),   # 10辆任务车，10辆服务车
        (20, 20),   # 20辆任务车，20辆服务车
        (25, 25),   # 25辆任务车，25辆服务车
        (30, 30),   # 30辆任务车，30辆服务车
        (35, 35),   # 35辆任务车，35辆服务车
        (40, 40),   # 40辆任务车，40辆服务车
    ]
    
    print(f"📋 计划测试 {len(vehicle_configs)} 种车辆配置: {vehicle_configs}")
    
    # 存储所有结果用于综合对比
    all_results = {}
    all_result_files = []
    
    for n_task, n_service in vehicle_configs:
        print(f"\n{'='*80}")
        print(f"🚗 测试配置: {n_task}辆任务车, {n_service}辆服务车")
        print(f"{'='*80}")
        
        # 初始化对比器
        comparator = OptimizedPerformanceComparator(
            n_task_vehicles=n_task,
            n_service_vehicles=n_service,
            mappo_model_path="k_value_mappo_final.pth",
            mappo_snr_model_path="mappo-snr/k_value_mappo_snr_final.pth",
            madqn_model_path=f"k_value_madqn_final_20vehicles_per_exponential.pth"
        )
        
        # 运行优化对比测试
        comparator.run_optimized_comparison(n_steps=100)  # 100个仿真步骤
        
        # 保存结果
        result_file = comparator.save_results()
        all_result_files.append(result_file)
        
        # 存储结果用于综合对比
        all_results[n_task] = comparator.results
        
        print(f"\n✅ {n_task}辆车的数据收集完成！")
    
    # 保存综合结果
    comprehensive_filename = f"comprehensive_results.pkl"
    comprehensive_data = {
        'all_results': all_results,
        'vehicle_configs': vehicle_configs,
        'result_files': all_result_files,
        'test_type': 'comprehensive_multi_vehicle'
    }
    
    with open(comprehensive_filename, 'wb') as f:
        pickle.dump(comprehensive_data, f)
    
    print(f"💾 综合结果已保存到: {comprehensive_filename}")
    print("🎉 数据收集完成！")


def plot_from_saved_data(data_file="comprehensive_results.pkl"):
    """从保存的数据文件中绘制图表"""
    print(f"📊 开始绘图模式，从文件加载数据: {data_file}")
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件 {data_file} 不存在！请先运行数据收集模式。")
        return
    
    # 加载数据
    try:
        with open(data_file, 'rb') as f:
            comprehensive_data = pickle.load(f)
        
        all_results = comprehensive_data['all_results']
        vehicle_configs = comprehensive_data['vehicle_configs']
        
        print(f"✅ 成功加载数据，包含 {len(all_results)} 种车辆配置")
        
    except Exception as e:
        print(f"❌ 加载数据文件失败: {e}")
        return
    
    # 绘制综合对比图
    print(f"\n{'='*80}")
    print("📈 生成综合对比图...")
    print(f"{'='*80}")
    
    multi_plot_path = f"multi_vehicle_comparison.png"
    OptimizedPerformanceComparator.plot_multi_vehicle_comparison(all_results, save_path=multi_plot_path)
    
    # 生成多车辆传输延迟对比图
    transmission_plot_path = f"multi_vehicle_transmission_comparison.png"
    OptimizedPerformanceComparator.plot_multi_vehicle_transmission_comparison(all_results, save_path=transmission_plot_path)
    
    # 生成数据量分布表格
    table_plot_path = f"data_distribution_table.png"
    OptimizedPerformanceComparator.plot_data_distribution_table(all_results, save_path=table_plot_path)
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("🎉 所有绘图完成！综合总结:")
    print(f"{'='*80}")
    
    for n_task in sorted(all_results.keys()):
        results = all_results[n_task]
        if results.get('mappo_snr_lp') and results.get('shannon'):
            mappo_delays = [r['total_delay_semantic'] for r in results['mappo_snr_lp']]
            # 添加异常值保护：过滤香农方案中大于3000ms的延迟值
            all_shannon_delays = [r['total_delay_shannon'] for r in results['shannon']]
            shannon_delays = [d for d in all_shannon_delays if d * 1000 <= 3000]
            if len(shannon_delays) == 0:  # 如果全部异常，使用原始数据
                shannon_delays = all_shannon_delays
                
            avg_mappo = np.mean(mappo_delays) * 1000  # 转换为毫秒
            avg_shannon = np.mean(shannon_delays) * 1000
            improvement = (avg_shannon - avg_mappo) / avg_shannon * 100
            
            print(f"📊 {n_task}辆车: MAPPO-SNR={avg_mappo:.1f}ms, Shannon={avg_shannon:.1f}ms, 改进={improvement:.1f}%")
    
    print("🎨 所有图表已生成完成！")


def main_menu():
    """主菜单，选择运行模式"""
    print("="*60)
    print("🚀 多车辆性能对比测试系统")
    print("="*60)
    print("请选择运行模式:")
    print("1. 收集数据模式 - 运行测试收集数据（耗时较长）")
    print("2. 绘图模式 - 从保存的数据绘制图表（快速）")
    print("3. 完整模式 - 收集数据并绘图（原始模式）")
    print("4. 单配置测试模式 - 快速测试单个配置")
    print("="*60)
    
    while True:
        choice = input("请输入选择 (1/2/3/4): ").strip()
        
        if choice == "1":
            collect_data_only()
            break
        elif choice == "2":
            plot_from_saved_data()
            break
        elif choice == "3":
            main()
            break
        elif choice == "4":
            test_single_config()
            break
        else:
            print("❌ 无效选择，请输入 1、2、3 或 4")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            # 单配置测试模式
            test_single_config()
        elif sys.argv[1] == "collect":
            # 数据收集模式
            collect_data_only()
        elif sys.argv[1] == "plot":
            # 绘图模式
            plot_from_saved_data()
        elif sys.argv[1] == "full":
            # 完整模式
            main()
        else:
            print("❌ 未知参数，支持的参数: single, collect, plot, full")
            main_menu()
    else:
        # 交互式菜单模式
        main_menu()
