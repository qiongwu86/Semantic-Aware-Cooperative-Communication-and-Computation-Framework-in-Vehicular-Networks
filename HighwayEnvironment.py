from __future__ import division
import numpy as np
import time
import random
import math
from Environment import Environ  # 导入Environment.py的环境类
from lambda_lp_solver import compute_optimal_lambda

class Vehicle:
    """车辆类"""
    def __init__(self, vehicle_id, position, velocity, vehicle_type):
        self.id = vehicle_id
        self.position = position  # [x, y]
        self.velocity = velocity  # m/s
        self.vehicle_type = vehicle_type  # 'task' or 'service'
        self.direction = 'r'  # 默认向右行驶，兼容Environment.py
        self.is_occupied = False  # 仅对服务车辆有效
        self.current_task = None  # 当前处理的任务
        self.served_vehicles = []  # 服务的任务车辆列表
        self.neighbors = []  # 兼容Environment.py
        self.destinations = []  # 兼容Environment.py
        
class Task:
    """任务类"""
    def __init__(self, task_id, source_vehicle, task_size, arrival_time):
        self.id = task_id
        self.source_vehicle = source_vehicle
        self.task_size = task_size  # Mbit
        self.data_size = task_size  # 兼容别名
        self.computation_size = task_size * 1000  # CPU cycles (假设1Mbit需要1000 cycles)
        self.arrival_time = arrival_time
        self.local_ratio = 0.1  # 本地执行比例，默认10%
        self.edge_ratio = 0.4   # 边缘服务器执行比例，默认40%
        self.bs_ratio = 0.5     # 基站执行比例，默认50%
        self.k_value = 4        # 语义传输参数k，按照论文默认值
        self.k_v2v = 4          # V2V链路的k值
        self.k_v2i = 4          # V2I链路的k值
        
class HighwayEnvironment(Environ):
    """高速公路环境类 - 继承自Environment.py的Environ类"""
    def __init__(self, n_task_vehicles=20, n_service_vehicles=5, enable_lp=True, random_seed=42, task_size=0.4):
        # 设置固定随机种子确保结果可重现
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # 高速公路特有参数
        self.road_length = 400  # 道路长度400m
        self.lane_width = 3.5   # 车道宽度3.5m
        self.lane_positions = [1.75, 5.25]  # 两个车道的中心位置
        
        # 建筑物遮挡参数（HighwayEnvironment特有功能）
        self.building_blockage_prob = 0.5  # 建筑物遮挡概率
        self.building_loss_mean = 20       # 建筑物遮挡损耗均值 (dB)
        self.building_loss_std = 8         # 建筑物遮挡损耗标准差 (dB)
        self.enable_building_loss = False  # 建筑物遮挡损耗控制标志
        
        # 时隙参数
        self.time_slot = 0.3  # 时隙长度100ms
        self.current_time = 0
        
        # 车辆参数
        self.n_task_vehicles = n_task_vehicles    # 任务车辆数量
        self.n_service_vehicles = n_service_vehicles  # 服务车辆数量
        self.n_vehicles = self.n_task_vehicles + self.n_service_vehicles  # 总车辆数
        self.enable_lp = enable_lp  # 是否启用线性规划求最优lambda
        
        # 参数验证
        if self.n_task_vehicles <= 0 or self.n_service_vehicles <= 0:
            raise ValueError("任务车辆和服务车辆数量都必须大于0")
        if self.n_vehicles > 60:  # 设置合理的上限
            raise ValueError("总车辆数不能超过60辆")
        
        # 初始化Environment基类
        down_lane = [i for i in range(self.road_length)]
        up_lane = [i for i in range(self.road_length)]
        left_lane = [1.75]  # 左车道位置
        right_lane = [5.25]  # 右车道位置
        
        # 调用父类初始化
        super().__init__(down_lane, up_lane, left_lane, right_lane, 
                        self.road_length, max(self.lane_positions) + 2)
        
        # 覆盖部分Environment参数以匹配我们的需求
        self.n_RB = 20  # 减少资源块强制复用
        self.V2V_power_dB = 15
        self.V2I_power_dB = 23
        self.n_Veh = self.n_vehicles  # 设置父类需要的车辆数量参数
        
        # 重新初始化信道对象以匹配我们的车辆数和资源块数
        self.V2Vchannels = self.V2Vchannels.__class__(self.n_vehicles, self.n_RB)
        self.V2Ichannels = self.V2Ichannels.__class__(self.n_vehicles, self.n_RB)
        
        # HighwayEnvironment特有的车辆管理
        self.task_vehicles = []
        self.service_vehicles = []
        
        # 资源块分配计数器（HighwayEnvironment特有）
        self.rb_allocation_counter = 0
        self.n_RB_per_link = 1  # 每个链路分配1个资源块
        
        # 带宽计算（HighwayEnvironment特有）
        self.bandwidth_per_RB = 540e3  # 每个资源块540kHz带宽
        self.bandwidth_per_link = self.n_RB_per_link * self.bandwidth_per_RB  # 每个链路540kHz
        self.total_bandwidth = self.n_RB * self.bandwidth_per_RB  # 总带宽
        
        # 计算参数
        self.local_cpu_freq = 1e9   # 本地CPU频率1GHz
        self.service_cpu_freq = 3e9 # 服务车辆CPU频率3GHz
        self.bs_cpu_freq = 6e9      # 基站CPU频率6GHz
        self.cpu_cycles_per_bit = 1000  # 每bit需要的CPU周期数
        
        # 任务参数
        self.poisson_rate = 4.0  # 泊松到达率，每秒每车辆平均4个任务
        self.task_size = task_size  # 固定任务大小（Mbit）
        self.task_counter = 0
        
        # 记录任务车辆与服务车辆的配对关系
        self.task_service_pairs = []  # [(task_vehicle_idx, service_vehicle_idx, rb_indices, snr_v2v, snr_v2i, k_value), ...]
        
        # 累积统计数据（用于最终统计和可视化）
        self.all_pairs_history = []  # 所有时隙的链路配对历史
        
        # 语义传输参数（严格按照论文设置）
        self.Aw = 20   # 平均单词长度每句
        self.Ah = 1200 # 平均硬件需求每句 (bits)
        self.default_k = 4  # 默认语义符号数每单词
        
        # 信道对象由父类Environment提供
        
        # 语义相似度查找表（从mat文件加载）
        self.semantic_table = self._load_semantic_table()
        
        self.initialize_vehicles()
    
    @classmethod
    def create_balanced_environment(cls, total_vehicles=20, task_size=0.4):
        """创建平衡的环境（任务车辆和服务车辆各一半）"""
        n_task = total_vehicles // 2
        n_service = total_vehicles - n_task
        return cls(n_task_vehicles=n_task, n_service_vehicles=n_service, task_size=task_size)
    
    @classmethod
    def create_task_heavy_environment(cls, total_vehicles=20, task_ratio=0.8, task_size=0.4):
        """创建任务车辆较多的环境"""
        n_task = int(total_vehicles * task_ratio)
        n_service = total_vehicles - n_task
        if n_service == 0:
            n_service = 1
            n_task = total_vehicles - 1
        return cls(n_task_vehicles=n_task, n_service_vehicles=n_service, task_size=task_size)
    
    @classmethod
    def create_service_heavy_environment(cls, total_vehicles=20, service_ratio=0.6, task_size=0.4):
        """创建服务车辆较多的环境"""
        n_service = int(total_vehicles * service_ratio)
        n_task = total_vehicles - n_service
        if n_task == 0:
            n_task = 1
            n_service = total_vehicles - 1
        return cls(n_task_vehicles=n_task, n_service_vehicles=n_service, task_size=task_size)
        
    def calculate_building_loss(self, pos1, pos2):
        """计算建筑物遮挡损耗 - 简化模型"""
        # 计算两点间距离
        distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        
        # 基于距离的确定性遮挡模型
        if distance < 50:
            # 近距离，很少遮挡
            blockage_factor = 0.1
        elif distance < 100:
            # 中距离，中等遮挡
            blockage_factor = 0.3
        elif distance < 200:
            # 远距离，较多遮挡
            blockage_factor = 0.6
        else:
            # 很远距离，大量遮挡
            blockage_factor = 0.8
        
        # 计算期望遮挡损耗（不使用随机性）
        expected_loss = self.building_loss_mean * blockage_factor
        
        # 添加一些随机变化
        random_variation = np.random.normal(0, self.building_loss_std * 0.3)
        total_loss = max(0, expected_loss + random_variation)
        
        return total_loss
        
    def _load_semantic_table(self):
        """从sem_table.mat文件加载语义相似度查找表"""
        try:
            from scipy.io import loadmat
            
            # 加载mat文件
            mat_data = loadmat('sem_table.mat')
            
            # 假设变量名为'sem_table'，格式为31*20
            # 行：SNR从-10到20dB (共31行)  
            # 列：k值从1到20 (共20列)
            if 'sem_table' in mat_data:
                semantic_matrix = mat_data['sem_table']
            else:
                # 尝试其他可能的变量名
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        semantic_matrix = mat_data[key]
                        break
            
            print(f"成功加载语义表，形状: {semantic_matrix.shape}")
            return semantic_matrix
            
        except ImportError:
            print("警告: scipy未安装，使用默认语义表")
            return self._create_default_semantic_table()
        except FileNotFoundError:
            print("警告: sem_table.mat文件未找到，使用默认语义表")
            return self._create_default_semantic_table()
        except Exception as e:
            print(f"警告: 加载sem_table.mat失败 ({e})，使用默认语义表")
            return self._create_default_semantic_table()
    
    def _create_default_semantic_table(self):
        """创建默认的语义相似度表（备用）"""
        # 创建31x20的矩阵 (SNR: -10到20dB, k: 1到20)
        snr_range = np.arange(-10, 21, 1)  # -10 to 20 dB (31个值)
        k_range = np.arange(1, 21, 1)      # 1 to 20 (20个值)
        
        table = np.zeros((len(snr_range), len(k_range)))
        
        for i, snr in enumerate(snr_range):
            for j, k in enumerate(k_range):
                # 简化的语义相似度函数
                delta = 1 / (1 + np.exp(-(snr + k - 5) / 10))
                table[i, j] = max(0.1, min(0.99, delta))
        
        return table
    
    def initialize_vehicles(self):
        """初始化车辆"""
        self.vehicles = []
        self.task_vehicles = []
        self.service_vehicles = []
        
        # 重新设置随机种子确保初始化的确定性
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # 生成车辆，按指定数量分配任务车辆和服务车辆
        for i in range(self.n_vehicles):
            # 随机选择车道（但种子已固定）
            lane_idx = random.randint(0, 1)
            y_pos = self.lane_positions[lane_idx]
            
            # 在道路范围内均匀分布（随机但可重现）
            x_pos = random.uniform(0, self.road_length)
            
            # 匀速行驶，速度在50-80km/h之间（随机但可重现）
            velocity = random.uniform(50/3.6, 80/3.6)  # 转换为m/s
            
            # 确定车辆类型：前n_task_vehicles个为任务车辆，其余为服务车辆
            vehicle_type = 'task' if i < self.n_task_vehicles else 'service'
            
            vehicle = Vehicle(i, [x_pos, y_pos], velocity, vehicle_type)
            self.vehicles.append(vehicle)
            
            if vehicle_type == 'task':
                self.task_vehicles.append(vehicle)
            else:
                self.service_vehicles.append(vehicle)
                
        print(f"初始化完成：{self.n_task_vehicles}辆任务车辆，{self.n_service_vehicles}辆服务车辆，总计{self.n_vehicles}辆车")
                
    def update_positions(self):
        """更新车辆位置"""
        for vehicle in self.vehicles:
            # 沿x方向移动
            vehicle.position[0] += vehicle.velocity * self.time_slot
            
            # 如果超出道路范围，则重新进入
            if vehicle.position[0] > self.road_length:
                vehicle.position[0] = 0
                

        
    def generate_tasks(self):
        """根据泊松分布生成任务"""
        new_tasks = []
        for vehicle in self.task_vehicles:
            # 泊松分布判断是否有任务到达（使用固定种子的随机数）
            if np.random.poisson(self.poisson_rate * self.time_slot) > 0:
                # 使用固定任务大小
                task_size = self.task_size
                
                # 创建任务
                task = Task(self.task_counter, vehicle, task_size, self.current_time)
                new_tasks.append(task)
                self.task_counter += 1
                
        return new_tasks
    
    def find_nearest_service_vehicle(self, task_vehicle):
        """找到最近的可用服务车辆"""
        min_distance = float('inf')
        selected_vehicle = None
        
        # 计算到所有服务车辆的距离
        distances = []
        for service_vehicle in self.service_vehicles:
            distance = math.hypot(
                task_vehicle.position[0] - service_vehicle.position[0],
                task_vehicle.position[1] - service_vehicle.position[1]
            )
            distances.append((distance, service_vehicle))
            
        # 按距离排序
        distances.sort(key=lambda x: x[0])
        
        # 找到最近的未占用服务车辆
        for distance, service_vehicle in distances:
            if not service_vehicle.is_occupied:
                return service_vehicle
                
        # 如果所有服务车辆都被占用，返回最近的服务车辆
        return distances[0][1]
    
    def _calculate_simple_snr_v2v(self, tx_idx, rx_idx, rb_idx):
        """简化的V2V SNR计算 - 直接使用Environment.py的信道数据"""
        # 使用Environment.py预计算的信道数据
        if hasattr(self, 'V2V_channels_with_fastfading'):
            # 计算建筑物损耗
            tx_pos = self.vehicles[tx_idx].position if hasattr(self, 'vehicles') and len(self.vehicles) > tx_idx else [0, 0]
            rx_pos = self.vehicles[rx_idx].position if hasattr(self, 'vehicles') and len(self.vehicles) > rx_idx else [0, 0]
            building_loss = self.calculate_building_loss(tx_pos, rx_pos) if self.enable_building_loss else 0
            
            # 信号功率
            signal_power_dB = (self.V2V_power_dB - self.V2V_channels_with_fastfading[tx_idx][rx_idx][rb_idx] - building_loss + 
                              2 * self.vehAntGain - self.vehNoiseFigure)
            
            # 简化：只考虑噪声，不考虑干扰（Environment.py会在后续计算中处理干扰）
            snr_dB = signal_power_dB - self.sig2_dB
            return snr_dB
        else:
            # 后备方案
            return 0.0
    
    def _calculate_simple_snr_v2i(self, vehicle_idx, rb_idx):
        """简化的V2I SNR计算 - 直接使用Environment.py的信道数据"""
        # 使用Environment.py预计算的信道数据
        if hasattr(self, 'V2I_channels_with_fastfading'):
            # 计算建筑物损耗
            vehicle_pos = self.vehicles[vehicle_idx].position if hasattr(self, 'vehicles') and len(self.vehicles) > vehicle_idx else [0, 0]
            bs_pos = [self.width/2, self.height/2]  # 使用Environment.py的基站位置
            building_loss = self.calculate_building_loss(vehicle_pos, bs_pos) if self.enable_building_loss else 0
            
            # 信号功率
            signal_power_dB = (self.V2I_power_dB - self.V2I_channels_with_fastfading[vehicle_idx][rb_idx] - building_loss + 
                              self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
            
            snr_dB = signal_power_dB - self.sig2_dB
            return snr_dB
        else:
            # 后备方案
            return 0.0
    
    def _calculate_v2v_sinr_with_interference(self, tx_idx, rx_idx, rb_indices):
        """计算包含链路间干扰的V2V SINR（单个资源块）"""
        # 现在每个链路只使用一个资源块
        rb_idx = rb_indices[0]
        
        tx_pos = self.vehicles[tx_idx].position
        rx_pos = self.vehicles[rx_idx].position
        building_loss = self.calculate_building_loss(tx_pos, rx_pos) if self.enable_building_loss else 0
        
        # V2V期望信号功率
        signal_power_dB = (self.V2V_power_dB - self.V2V_channels_with_fastfading[tx_idx][rx_idx][rb_idx] - building_loss + 
                          2 * self.vehAntGain - self.vehNoiseFigure)
        signal_power_linear = 10**(signal_power_dB/10)
        
        # 干扰功率
        interference_power = self.sig2  # 噪声功率
        
        # 1. V2I链路对V2V的干扰（如果V2I也使用这个资源块）
        bs_pos = [self.width/2, self.height/2]
        v2i_building_loss = self.calculate_building_loss(bs_pos, rx_pos) if self.enable_building_loss else 0
        v2i_interference_dB = (self.V2I_power_dB - self.V2I_channels_abs[0] - v2i_building_loss + 
                              self.bsAntGain + self.vehAntGain - self.vehNoiseFigure)
        interference_power += 10**(v2i_interference_dB/10)
        
        # 2. 其他V2V链路在同一资源块上的干扰（只考虑当前时隙）
        interference_count = 0
        for other_tx_idx, other_rx_idx, other_rb_list, _, _, _, _ in self.task_service_pairs:
            if (other_tx_idx != tx_idx and len(other_rb_list) > 0 and 
                rb_idx in other_rb_list):  # 同一资源块
                
                # 其他V2V发送方对当前接收方的干扰
                other_tx_pos = self.vehicles[other_tx_idx].position
                other_building_loss = self.calculate_building_loss(other_tx_pos, rx_pos) if self.enable_building_loss else 0
                
                other_interference_dB = (self.V2V_power_dB - 
                                       self.V2V_channels_with_fastfading[other_tx_idx][rx_idx][rb_idx] - 
                                       other_building_loss + 2 * self.vehAntGain - self.vehNoiseFigure)
                interference_power += 10**(other_interference_dB/10)
                interference_count += 1
        
        # 计算SINR
        sinr_linear = signal_power_linear / interference_power if interference_power > 0 else 0
        sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
        
        return sinr_dB
    
    def _calculate_v2i_sinr_with_interference(self, tx_idx, rb_indices):
        """计算包含链路间干扰的V2I SINR（单个资源块）"""
        # 现在每个链路只使用一个资源块
        rb_idx = rb_indices[0]
        
        tx_pos = self.vehicles[tx_idx].position
        bs_pos = [self.width/2, self.height/2]
        building_loss = self.calculate_building_loss(tx_pos, bs_pos) if self.enable_building_loss else 0
        
        # V2I期望信号功率
        signal_power_dB = (self.V2I_power_dB - self.V2I_channels_abs[tx_idx] - building_loss + 
                          self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
        signal_power_linear = 10**(signal_power_dB/10)
        
        # 干扰功率
        interference_power = self.sig2  # 噪声功率
        
        # 同一资源块上其他V2V链路对基站的干扰（只考虑当前时隙）
        interference_count = 0
        for other_tx_idx, other_rx_idx, other_rb_list, _, _, _, _ in self.task_service_pairs:
            if (other_tx_idx != tx_idx and len(other_rb_list) > 0 and 
                rb_idx in other_rb_list):  # 同一资源块
                
                # 其他V2V发送方对基站的干扰
                other_tx_pos = self.vehicles[other_tx_idx].position
                other_building_loss = self.calculate_building_loss(other_tx_pos, bs_pos) if self.enable_building_loss else 0
                
                # V2V链路对V2I的上行干扰（使用V2I信道模型）
                other_interference_dB = (self.V2V_power_dB - 
                                       self.V2I_channels_abs[other_tx_idx] - 
                                       other_building_loss + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
                interference_power += 10**(other_interference_dB/10)
                interference_count += 1
        
        # 计算SINR
        sinr_linear = signal_power_linear / interference_power if interference_power > 0 else 0
        sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
        
        return sinr_dB

    

    
    def select_k_value(self, snr_dB):
        """根据信噪比自适应选择k值"""
        if snr_dB > 20:
            return 1  # 最小k值为1，避免除零错误
        elif snr_dB < -10:
            return 20
        else:
            # 在-10到20dB范围内，使用默认k值
            return self.default_k
    
    def get_semantic_similarity(self, snr_dB, k_value=None):
        """获取语义相似度"""
        # 自适应k值选择
        if k_value is None:
            k_value = self.select_k_value(snr_dB)
        
        # 特殊情况处理
        if snr_dB > 20:
            return 0.9
        elif snr_dB < -10:
            return 0.8
        
        # 从查找表获取语义相似度（-10到20dB范围）
        #try:
            # 将SNR映射到表格索引 (-10dB对应索引0, 20dB对应索引30)
        snr_idx = int(round(snr_dB + 10))
        snr_idx = np.clip(snr_idx, 0, 30)  # 确保索引在有效范围内
            
            # 将k值映射到表格索引（k值1-20对应索引0-19）
        k_idx = int(round(k_value - 1))  # k=1对应索引0, k=20对应索引19
        k_idx = np.clip(k_idx, 0, 19)  # 确保k索引在有效范围内
            
            # 从语义表获取相似度
        similarity = self.semantic_table[k_idx, snr_idx]
        return float(similarity)
            
        #except (IndexError, TypeError):
            ## 如果查找表访问失败，使用备用计算
            #delta = 1 / (1 + np.exp(-(snr_dB + k_value - 5) / 10))
            #return max(0.1, min(0.99, delta))
    
    def calculate_shannon_rate_and_delay(self, snr_dB, task_size_mbit):
        """计算基于香农容量的传输速率和时延"""
        # 转换信噪比到线性值
        snr_linear = 10**(snr_dB/10)
        
        # 香农容量（使用每个链路的总带宽：2个资源块）
        capacity = self.bandwidth_per_link * np.log2(1 + snr_linear)  # bps
        
        # 传输时延
        task_size_bits = task_size_mbit * 1e6  # Mbit转换为bits
        delay = task_size_bits / capacity  # 秒
        
        return capacity, delay
    
    def calculate_semantic_rate_and_delay(self, snr_dB, k_value, task_size_mbit):
        """计算基于语义传输的速率和时延 - 使用自适应k值"""
        # 如果k_value为None，则自动选择最优k值
        if k_value is None:
            k_value = self.select_k_value(snr_dB)
        
        # 确保k值有效（>=1）
        k_value = max(1, k_value)
        
        # 获取语义相似度
        delta = self.get_semantic_similarity(snr_dB, k_value)
        
        # 确保delta有效（>0）
        delta = max(0.001, delta)
        
        # 将任务大小从Mbit转换为bits
        task_size_bits = task_size_mbit * 1e6
        
        # 计算句子数量：任务大小(bits) / 每句硬件需求(bits)
        num_sentences = task_size_bits / self.Ah
        
        # 根据论文，As[t]代表平均语义信息每句，这里我们假设As = num_sentences
        # 因为As应该与任务的语义内容相关
        As = num_sentences
        
        # 公式3: 语义传输速率 Ri[t] = B*As[t] / (Aw[t]*ki[t]*δi[t])
        semantic_rate = (self.bandwidth_per_link * As) / (self.Aw * k_value * delta)  # suts/s
        
        # 公式5: 传输时延 T_Tr_i[t] = λi[t]*di[t]*ki[t]*Aw[t] / (B*δi[t])
        # 这里λi[t]是任务比例因子，di[t]是任务队列数
        # 简化假设：λi[t] = 1, di[t] = num_sentences
        lambda_i = 1.0  # 任务比例因子
        di = num_sentences  # 任务队列数（句子数）
        
        delay = (lambda_i * di * k_value * self.Aw) / (self.bandwidth_per_link * delta)
        
        return semantic_rate, delay
    
    def calculate_computation_delay(self, task_size_mbit, cpu_freq, num_parallel_tasks=1):
        """计算计算时延"""
        task_size_bits = task_size_mbit * 1e6
        total_cycles = task_size_bits * self.cpu_cycles_per_bit
        
        # 如果有多个任务并行，平分CPU频率
        effective_freq = cpu_freq / num_parallel_tasks
        
        delay = total_cycles / effective_freq
        return delay
    
    def process_task(self, task, num_tasks_current_slot=1):
        """处理单个任务"""
        task_vehicle = task.source_vehicle
        task_vehicle_idx = self.vehicles.index(task_vehicle)
        
        # 找到最近的服务车辆
        service_vehicle = self.find_nearest_service_vehicle(task_vehicle)
        service_vehicle_idx = self.vehicles.index(service_vehicle)
        
        # 标记服务车辆占用状态
        if not service_vehicle.is_occupied:
            service_vehicle.is_occupied = True
            service_vehicle.served_vehicles = [task_vehicle]
        else:
            service_vehicle.served_vehicles.append(task_vehicle)
        
        # 分配资源块（每个链路2个资源块）
        rb_indices = self.allocate_resource_blocks()
        
        # 先临时记录配对关系（用于干扰计算）
        temp_pair = (task_vehicle_idx, service_vehicle_idx, rb_indices, 0, 0, 0, 0)
        self.task_service_pairs.append(temp_pair)
        
        # 使用HighwayEnvironment自己的干扰计算
        # 更新信道状态
        self.renew_channel()
        self.renew_channels_fastfading()
        
        # 计算包含链路间干扰的SINR（基于所有分配的资源块）
        snr_v2v = self._calculate_v2v_sinr_with_interference(task_vehicle_idx, service_vehicle_idx, rb_indices)
        snr_v2i = self._calculate_v2i_sinr_with_interference(task_vehicle_idx, rb_indices)
        
        # 分别为V2V和V2I链路选择k值
        k_v2v = self.select_k_value(snr_v2v)
        k_v2i = self.select_k_value(snr_v2i)
        

        
        # 更新最后一个记录的完整信息 - 现在包含V2V和V2I的k值
        self.task_service_pairs[-1] = (task_vehicle_idx, service_vehicle_idx, rb_indices, snr_v2v, snr_v2i, k_v2v, k_v2i)
        
        # 同时保存到累积历史记录中（用于统计和可视化）
        self.all_pairs_history.append((task_vehicle_idx, service_vehicle_idx, rb_indices, snr_v2v, snr_v2i, k_v2v, k_v2i))
        
        # 基于固定k的线性规划（凸优化）计算最优lambda（可开关）
        full_task_size = task.task_size
        num_tasks_on_service = len(service_vehicle.served_vehicles)
        lp_t_opt = None
        if getattr(self, 'enable_lp', True):
            # A_local, A_edge, A_bs: 满载(λ=1)时每分支延迟（语义版本 + 计算延迟），单位秒
            local_A = self.calculate_computation_delay(full_task_size, self.local_cpu_freq)
            _, v2v_sem_delay_full = self.calculate_semantic_rate_and_delay(snr_v2v, k_v2v, full_task_size)
            service_comp_full = self.calculate_computation_delay(full_task_size, self.service_cpu_freq, num_tasks_on_service)
            edge_A = v2v_sem_delay_full + service_comp_full
            _, v2i_sem_delay_full = self.calculate_semantic_rate_and_delay(snr_v2i, k_v2i, full_task_size)
            # 计算基站满载延迟时，使用当前时隙的任务数量
            bs_comp_full = self.calculate_computation_delay(full_task_size, self.bs_cpu_freq, num_tasks_current_slot)
            bs_A = v2i_sem_delay_full + bs_comp_full

            lambda_local, lambda_edge, lambda_bs, lp_t_opt = compute_optimal_lambda(local_A, edge_A, bs_A, prefer="closed_form")

            # 更新任务的比例（供记录/后续计算）
            task.local_ratio = lambda_local
            task.edge_ratio = lambda_edge
            task.bs_ratio = lambda_bs
        else:
            lambda_local = task.local_ratio
            lambda_edge = task.edge_ratio
            lambda_bs = task.bs_ratio

        # 计算各部分任务大小（按当前比例）
        local_task_size = full_task_size * lambda_local
        edge_task_size = full_task_size * lambda_edge
        bs_task_size = full_task_size * lambda_bs
        
        # 1. 本地计算时延
        local_delay = self.calculate_computation_delay(local_task_size, self.local_cpu_freq)
        
        # 2. V2V + 服务车辆计算时延
        # V2V传输时延（香农）
        _, v2v_shannon_delay = self.calculate_shannon_rate_and_delay(snr_v2v, edge_task_size)
        # V2V传输时延（语义）
        _, v2v_semantic_delay = self.calculate_semantic_rate_and_delay(snr_v2v, k_v2v, edge_task_size)
        
        # 服务车辆计算时延
        service_comp_delay = self.calculate_computation_delay(edge_task_size, self.service_cpu_freq, num_tasks_on_service)
        
        v2v_total_delay_shannon = v2v_shannon_delay + service_comp_delay
        v2v_total_delay_semantic = v2v_semantic_delay + service_comp_delay
        
        # 3. V2I + 基站计算时延
        # V2I传输时延（香农）
        _, v2i_shannon_delay = self.calculate_shannon_rate_and_delay(snr_v2i, bs_task_size)
        # V2I传输时延（语义）
        _, v2i_semantic_delay = self.calculate_semantic_rate_and_delay(snr_v2i, k_v2i, bs_task_size)
        
        # 基站计算时延（根据当前时隙的任务数平分资源）
        bs_comp_delay = self.calculate_computation_delay(bs_task_size, self.bs_cpu_freq, num_tasks_current_slot)
        
        v2i_total_delay_shannon = v2i_shannon_delay + bs_comp_delay
        v2i_total_delay_semantic = v2i_semantic_delay + bs_comp_delay
        
        # 总时延取最大值（三部分并行执行）
        total_delay_shannon = max(local_delay, v2v_total_delay_shannon, v2i_total_delay_shannon)
        total_delay_semantic = max(local_delay, v2v_total_delay_semantic, v2i_total_delay_semantic)
        
        return {
            'task_id': task.id,
            'task_vehicle_id': task_vehicle.id,
            'service_vehicle_id': service_vehicle.id,
            'snr_v2v': snr_v2v,
            'snr_v2i': snr_v2i,
            'local_delay': local_delay,
            'v2v_delay_shannon': v2v_total_delay_shannon,
            'v2v_delay_semantic': v2v_total_delay_semantic,
            'v2i_delay_shannon': v2i_total_delay_shannon,
            'v2i_delay_semantic': v2i_total_delay_semantic,
            'total_delay_shannon': total_delay_shannon,
            'total_delay_semantic': total_delay_semantic,
            'k_v2v': k_v2v,
            'k_v2i': k_v2i,
            'lambda_local': lambda_local,
            'lambda_edge': lambda_edge,
            'lambda_bs': lambda_bs,
            'lp_t_opt': lp_t_opt,
            'service_vehicle_tasks': num_tasks_on_service,  # 该服务车辆的任务数
            'total_tasks_in_slot': num_tasks_current_slot   # 当前时隙总任务数
        }
    
    def reset_service_vehicles(self):
        """重置服务车辆占用状态"""
        for vehicle in self.service_vehicles:
            vehicle.is_occupied = False
            vehicle.served_vehicles = []
    
    def allocate_resource_blocks(self):
        """循环分配资源块（每次分配1个资源块）"""
        # 直接循环分配单个资源块
        rb_idx = self.rb_allocation_counter % self.n_RB
        
        self.rb_allocation_counter += 1
        return [rb_idx]  # 返回包含单个资源块的列表，保持接口一致
    
    def collect_snr_statistics(self):
        """收集已记录的任务链路SNR统计"""
        v2v_snr_list = []
        v2i_snr_list = []
        rb_usage = {}
        
        # 从累积的历史记录中提取SNR数据和k值
        k_v2v_values = []
        k_v2i_values = []
        for task_idx, service_idx, rb_indices, snr_v2v, snr_v2i, k_v2v, k_v2i in self.all_pairs_history:
            v2v_snr_list.append(snr_v2v)
            v2i_snr_list.append(snr_v2i)
            k_v2v_values.append(k_v2v)
            k_v2i_values.append(k_v2i)
            
            # 统计资源块使用情况
            for rb_idx in rb_indices:
                if rb_idx not in rb_usage:
                    rb_usage[rb_idx] = 0
                rb_usage[rb_idx] += 1
        
        return {
            'v2v_snr': v2v_snr_list,
            'v2i_snr': v2i_snr_list,
            'k_v2v_values': k_v2v_values,
            'k_v2i_values': k_v2i_values,
            'rb_usage': rb_usage
        }
    
    def step(self):
        """执行一个时隙的仿真步骤"""
        # 1. 更新车辆位置
        self.update_positions()
        
        # 2. 更新信道状态 - 使用Environment.py的方法
        # 这将由Environment.py在需要时自动处理
        
        # 3. 重置服务车辆状态和链路配对信息
        self.reset_service_vehicles()
        self.task_service_pairs = []  # 重置当前时隙的链路配对信息
        
        # 4. 生成新任务
        new_tasks = self.generate_tasks()
        
        # 5. 处理任务 - 基站资源在当前时隙的所有任务间平分
        results = []
        num_tasks_current_slot = len(new_tasks)  # 当前时隙的任务数量
        
        # 先处理所有任务，建立任务-服务车辆的绑定关系
        for task in new_tasks:
            result = self.process_task(task, num_tasks_current_slot)
            results.append(result)
        
        # 6. 修正每个任务的服务车辆任务数为该服务车辆在本时隙的最终任务数
        for result in results:
            service_vehicle_id = result['service_vehicle_id']
            # 找到对应的服务车辆
            service_vehicle = next(v for v in self.service_vehicles if v.id == service_vehicle_id)
            # 更新为该服务车辆在本时隙的最终任务数
            result['service_vehicle_tasks'] = len(service_vehicle.served_vehicles)
        
        # 7. 更新时间
        self.current_time += self.time_slot
        
        return results
    
    def run_simulation(self, num_steps):
        """运行仿真"""
        all_results = []
        for step in range(num_steps):
            results = self.step()
            if results:
                all_results.extend(results)
            
            if step % 10 == 0:
                print(f"Step {step}, Generated {len(results)} tasks")
        
        return all_results

def test_performance_analysis(n_task_vehicles=30, n_service_vehicles=8, num_steps=100, enable_building_loss=False, task_size=0.4):
    """核心性能分析测试"""
    print("=" * 60)
    print(f"性能分析测试 - {n_task_vehicles}任务车辆, {n_service_vehicles}服务车辆")
    print(f"建筑物遮挡损耗: {'启用' if enable_building_loss else '禁用'}")
    print("=" * 60)
    
    # 创建环境
    env = HighwayEnvironment(n_task_vehicles=n_task_vehicles, n_service_vehicles=n_service_vehicles, random_seed=42, task_size=task_size)
    env.enable_building_loss = enable_building_loss  # 设置建筑物损耗标志
    
    # 运行仿真
    print(f"运行{num_steps}个时隙的仿真...")
    results = env.run_simulation(num_steps)
    
    if not results:
        print("仿真期间无任务生成")
        return
    
    # 收集SNR数据
    snr_data = env.collect_snr_statistics()
    v2v_snr = np.array(snr_data['v2v_snr'])
    v2i_snr = np.array(snr_data['v2i_snr'])
    
    # 分析时延性能
    shannon_delays = [r['total_delay_shannon'] for r in results]
    semantic_delays = [r['total_delay_semantic'] for r in results]
    v2v_delays_shannon = [r['v2v_delay_shannon'] for r in results]
    v2v_delays_semantic = [r['v2v_delay_semantic'] for r in results]
    v2i_delays_shannon = [r['v2i_delay_shannon'] for r in results]
    v2i_delays_semantic = [r['v2i_delay_semantic'] for r in results]
    
    print(f"\n=== 性能统计结果 ===")
    print(f"处理任务总数: {len(results)}")
    print(f"平均任务生成率: {len(results)/num_steps:.2f} 任务/时隙")
    
    print(f"\n=== SNR统计 ===")
    print(f"V2V SNR: 平均 {np.mean(v2v_snr):.2f} dB, 标准差 {np.std(v2v_snr):.2f} dB")
    print(f"V2I SNR: 平均 {np.mean(v2i_snr):.2f} dB, 标准差 {np.std(v2i_snr):.2f} dB")
    
    # k值统计 - 分别统计V2V和V2I
    k_v2v_values = snr_data.get('k_v2v_values', [])
    k_v2i_values = snr_data.get('k_v2i_values', [])
    
    if k_v2v_values and k_v2i_values:
        print(f"\n=== k值选择统计 ===")
        
        # V2V k值统计
        k_v2v_array = np.array(k_v2v_values)
        print(f"V2V k值: 平均 {np.mean(k_v2v_array):.2f}, 标准差 {np.std(k_v2v_array):.2f}")
        unique_k_v2v, counts_v2v = np.unique(k_v2v_array, return_counts=True)
        print("V2V k值分布:")
        for k, count in zip(unique_k_v2v, counts_v2v):
            percentage = count / len(k_v2v_array) * 100
            print(f"  k={k}: {count} ({percentage:.1f}%)")
        
        # V2I k值统计
        k_v2i_array = np.array(k_v2i_values)
        print(f"V2I k值: 平均 {np.mean(k_v2i_array):.2f}, 标准差 {np.std(k_v2i_array):.2f}")
        unique_k_v2i, counts_v2i = np.unique(k_v2i_array, return_counts=True)
        print("V2I k值分布:")
        for k, count in zip(unique_k_v2i, counts_v2i):
            percentage = count / len(k_v2i_array) * 100
            print(f"  k={k}: {count} ({percentage:.1f}%)")
    
    print(f"\n=== 时延统计 ===")
    print(f"总时延(香农): 平均 {np.mean(shannon_delays)*1000:.2f} ms, 标准差 {np.std(shannon_delays)*1000:.2f} ms")
    print(f"总时延(语义): 平均 {np.mean(semantic_delays)*1000:.2f} ms, 标准差 {np.std(semantic_delays)*1000:.2f} ms")
    print(f"V2V时延(香农): 平均 {np.mean(v2v_delays_shannon)*1000:.2f} ms")
    print(f"V2V时延(语义): 平均 {np.mean(v2v_delays_semantic)*1000:.2f} ms")
    print(f"V2I时延(香农): 平均 {np.mean(v2i_delays_shannon)*1000:.2f} ms")
    print(f"V2I时延(语义): 平均 {np.mean(v2i_delays_semantic)*1000:.2f} ms")
    
    # 计算传输速率
    shannon_rates = []
    semantic_rates = []
    for result in results:
        task_size = 0.4  # 假设平均任务大小
        shannon_rate, _ = env.calculate_shannon_rate_and_delay(result['snr_v2v'], task_size)
        semantic_rate, _ = env.calculate_semantic_rate_and_delay(result['snr_v2v'], 4, task_size)
        shannon_rates.append(shannon_rate)
        semantic_rates.append(semantic_rate)
    
    print(f"\n=== 传输速率统计 ===")
    print(f"香农容量: 平均 {np.mean(shannon_rates)/1e6:.2f} Mbps, 标准差 {np.std(shannon_rates)/1e6:.2f} Mbps")
    print(f"语义传输: 平均 {np.mean(semantic_rates):.2f} suts/s, 标准差 {np.std(semantic_rates):.2f} suts/s")
    
    # 计算性能增益
    delay_improvement = (np.mean(shannon_delays) - np.mean(semantic_delays)) / np.mean(shannon_delays) * 100
    print(f"\n=== 性能增益 ===")
    print(f"语义传输时延改善: {delay_improvement:.1f}%")
    
    # 绘制SNR分布图
    try:
        import matplotlib.pyplot as plt
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义5dB间隔的区间
        snr_bins = np.arange(-20, 35, 5)
        bin_labels = [f"{snr_bins[i]}-{snr_bins[i+1]}" for i in range(len(snr_bins)-1)]
        
        # 统计各区间的链路数量
        v2v_counts, _ = np.histogram(v2v_snr, bins=snr_bins)
        v2i_counts, _ = np.histogram(v2i_snr, bins=snr_bins)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(bin_labels))
        width = 0.35
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, v2v_counts, width, label='V2V Links', alpha=0.8)
        bars2 = ax.bar(x + width/2, v2i_counts, width, label='V2I Links', alpha=0.8)
        
        # 设置图表属性
        ax.set_xlabel('SNR Range (dB)')
        ax.set_ylabel('Number of Links')
        ax.set_title('SNR Distribution of V2V and V2I Links (5dB intervals)')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 在柱子上显示数值
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('snr_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n图表已保存为 snr_distribution.png")
        
    except ImportError:
        print("\n注意: matplotlib未安装，无法生成图表")
    
    return {
        'total_tasks': len(results),
        'v2v_snr_avg': np.mean(v2v_snr),
        'v2i_snr_avg': np.mean(v2i_snr),
        'shannon_delay_avg': np.mean(shannon_delays),
        'semantic_delay_avg': np.mean(semantic_delays),
        'shannon_rate_avg': np.mean(shannon_rates),
        'semantic_rate_avg': np.mean(semantic_rates),
        'delay_improvement': delay_improvement
    }

if __name__ == "__main__":
    # 核心性能测试 - 默认配置（关闭建筑物损耗）
    print("=== 测试1: 关闭建筑物遮挡损耗 ===")
    test_performance_analysis(enable_building_loss=False)
    
    # 可以测试不同配置
    print("\n" + "="*60)
    print("=== 测试2: 不同车辆数量配置（关闭建筑物损耗）===")
    test_performance_analysis(n_task_vehicles=20, n_service_vehicles=8, num_steps=50, enable_building_loss=False)
    
