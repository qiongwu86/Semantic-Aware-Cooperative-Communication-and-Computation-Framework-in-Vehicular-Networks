"""
MADQN奖励变化趋势绘制工具

从保存的训练数据文件中读取奖励数据，绘制奖励随训练回合数的变化趋势
基于k_value_madqn.py中的绘图代码
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def load_training_data(filename):
    """加载训练数据文件"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Training data file {filename} not found!")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data


def plot_reward_trends(data, save_path=None, method_type='MADQN'):
    """绘制奖励变化趋势（适合论文使用的紧凑格式）"""
    
    # 检查数据是否包含奖励信息
    if 'total_rewards' not in data or not data['total_rewards']:
        print("No reward data found in the training data!")
        return
    
    total_rewards = data['total_rewards']
    episodes = range(1, len(total_rewards) + 1)
    
    # 设置论文级别的字体和样式
    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 1.0,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif'
    })
    
    # 创建适合双栏论文的图形尺寸 (更方的比例)
    plt.figure(figsize=(6, 4.5))  # 双栏论文标准比例，更方一些
    
    # 根据方法类型选择颜色，与test文件一致的配色方案
    color_map = {
        'MAPPO-SNR': '#FFD700',  # 金色 - MAPPO-SNR专用，高显著性
        'MAPPO': '#2E86C1',      # 蓝色系 
        'MADQN': '#1ABC9C'       # 青色系
    }
    
    # 自动检测方法类型
    if 'snr' in save_path.lower() if save_path else False:
        method_type = 'MAPPO-SNR'
    elif 'mappo' in save_path.lower() if save_path else False:
        method_type = 'MAPPO'
    
    primary_color = color_map.get(method_type, '#1ABC9C')  # 默认使用MADQN颜色
    
    # 原始奖励曲线（半透明）
    plt.plot(episodes, total_rewards, alpha=0.3, color=primary_color, 
             linewidth=0.5, label='Episode Rewards')
    
    # 添加移动平均线（主要显示的曲线）
    if len(total_rewards) > 10:
        window_size = min(50, len(total_rewards) // 10)
        if window_size > 1:
            smoothed_rewards = []
            for i in range(window_size, len(total_rewards)):
                smoothed_rewards.append(np.mean(total_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(total_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_rewards, 
                    color=primary_color, linewidth=2.5, 
                    label=f'Moving Average (window={window_size})')
    
    # 论文风格的标题和标签
    plt.title(f'{method_type} Training Reward Convergence', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Training Episode', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    
    # 与test文件一致的网格样式
    plt.grid(True, alpha=0.3, axis='y')
    
    # 图例放到右上角，与test文件风格一致
    plt.legend(loc='upper right', fontsize=14, framealpha=0.95)
    
    # 调整边距以节省空间
    plt.tight_layout(pad=1.0)
    
    # 保存图形（高质量，适合论文）
    if save_path:
        # 保存为多种格式
        base_name = save_path.rsplit('.', 1)[0]
        
        # PNG格式（高分辨率）
        plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # PDF格式（矢量图，适合论文）
        plt.savefig(f"{base_name}.pdf", bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Reward trends plot saved as:")
        print(f"  - {base_name}.png (high-res raster)")
        print(f"  - {base_name}.pdf (vector format)")
    
    plt.show()
    
    # 打印简化的统计信息
    print("=" * 50)
    print(f"{method_type} Training Summary")
    print("=" * 50)
    print(f"Total Episodes: {len(total_rewards)}")
    print(f"Final Reward: {total_rewards[-1]:.4f}")
    print(f"Best Reward: {max(total_rewards):.4f} (Episode {total_rewards.index(max(total_rewards)) + 1})")
    print(f"Mean ± Std: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    
    # 收敛性分析
    if len(total_rewards) >= 100:
        last_100 = total_rewards[-100:]
        first_100 = total_rewards[:100]
        improvement = np.mean(last_100) - np.mean(first_100)
        print(f"Improvement (first 100 → last 100): {improvement:+.4f}")
    
    print("=" * 50)


def plot_three_way_comparison(madqn_data, mappo_data, mappo_snr_data, save_path=None):
    """绘制MADQN、MAPPO和MAPPO-SNR的三方奖励对比趋势（适合论文使用）"""
    
    # 检查数据
    if 'total_rewards' not in madqn_data or not madqn_data['total_rewards']:
        print("No MADQN reward data found!")
        return
    if 'total_rewards' not in mappo_data or not mappo_data['total_rewards']:
        print("No MAPPO reward data found!")
        return
    if 'total_rewards' not in mappo_snr_data or not mappo_snr_data['total_rewards']:
        print("No MAPPO-SNR reward data found!")
        return
    
    madqn_rewards = madqn_data['total_rewards']
    mappo_rewards = mappo_data['total_rewards']
    mappo_snr_rewards = mappo_snr_data['total_rewards']
    
    # 设置论文级别的字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'font.family': 'serif'
    })
    
    # 创建适合双栏论文的图形尺寸
    plt.figure(figsize=(7, 5))  # 稍微增大以容纳三条曲线
    
    # 使用与test文件一致的配色方案
    mappo_snr_color = '#FFD700'  # 金色 - MAPPO-SNR专用，高显著性（与test文件一致）
    madqn_color = '#1ABC9C'      # 青色系 (与test文件中madqn_lp颜色一致)
    mappo_color = '#2E86C1'      # 蓝色系 (与test文件中mappo_lp颜色一致)
    
    # MAPPO-SNR奖励曲线（放在首位，高亮显示）
    mappo_snr_episodes = range(1, len(mappo_snr_rewards) + 1)
    plt.plot(mappo_snr_episodes, mappo_snr_rewards, alpha=0.4, color=mappo_snr_color, 
             linewidth=0.8, label='Our-Method Episode Rewards')
    
    # MAPPO-SNR移动平均
    if len(mappo_snr_rewards) > 10:
        window_size = min(50, len(mappo_snr_rewards) // 10)
        if window_size > 1:
            smoothed_mappo_snr = []
            for i in range(window_size, len(mappo_snr_rewards)):
                smoothed_mappo_snr.append(np.mean(mappo_snr_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(mappo_snr_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_mappo_snr, 
                    color=mappo_snr_color, linewidth=3.0, label='Our-Method Moving Average')
    
    # MADQN奖励曲线
    madqn_episodes = range(1, len(madqn_rewards) + 1)
    plt.plot(madqn_episodes, madqn_rewards, alpha=0.3, color=madqn_color, 
             linewidth=0.5, label='MADQN Episode Rewards')
    
    # MADQN移动平均
    if len(madqn_rewards) > 10:
        window_size = min(50, len(madqn_rewards) // 10)
        if window_size > 1:
            smoothed_madqn = []
            for i in range(window_size, len(madqn_rewards)):
                smoothed_madqn.append(np.mean(madqn_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(madqn_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_madqn, 
                    color=madqn_color, linewidth=2.5, label='MADQN Moving Average')
    
    # MAPPO奖励曲线
    mappo_episodes = range(1, len(mappo_rewards) + 1)
    plt.plot(mappo_episodes, mappo_rewards, alpha=0.3, color=mappo_color, 
             linewidth=0.5, label='MAPPO Episode Rewards')
    
    # MAPPO移动平均
    if len(mappo_rewards) > 10:
        window_size = min(50, len(mappo_rewards) // 10)
        if window_size > 1:
            smoothed_mappo = []
            for i in range(window_size, len(mappo_rewards)):
                smoothed_mappo.append(np.mean(mappo_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(mappo_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_mappo, 
                    color=mappo_color, linewidth=2.5, label='MAPPO Moving Average')
    
    # 论文风格的标题和标签
    plt.title('Our-Method vs MAPPO vs MADQN Training Reward Convergence', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    
    # 与test文件一致的网格样式
    plt.grid(True, alpha=0.3, axis='y')
    
    # 图例放到右上角，与test文件风格一致
    plt.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # 调整边距以节省空间
    plt.tight_layout(pad=1.0)
    
    # 保存图形（高质量，适合论文）
    if save_path:
        # 保存为多种格式
        base_name = save_path.rsplit('.', 1)[0]
        
        # PNG格式（高分辨率）
        plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # PDF格式（矢量图，适合论文）
        plt.savefig(f"{base_name}.pdf", bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Three-way comparison plot saved as:")
        print(f"  - {base_name}.png (high-res raster)")
        print(f"  - {base_name}.pdf (vector format)")
    
    plt.show()
    
    # 打印三方对比统计信息
    print("=" * 70)
    print("MAPPO-SNR vs MAPPO vs MADQN Training Comparison")
    print("=" * 70)
    print(f"MAPPO-SNR - Episodes: {len(mappo_snr_rewards)}, Final: {mappo_snr_rewards[-1]:.4f}, Mean: {np.mean(mappo_snr_rewards):.4f}")
    print(f"MAPPO     - Episodes: {len(mappo_rewards)}, Final: {mappo_rewards[-1]:.4f}, Mean: {np.mean(mappo_rewards):.4f}")
    print(f"MADQN     - Episodes: {len(madqn_rewards)}, Final: {madqn_rewards[-1]:.4f}, Mean: {np.mean(madqn_rewards):.4f}")
    print("=" * 70)


def plot_comparison_reward_trends(madqn_data, mappo_data, save_path=None):
    """绘制MADQN和MAPPO的奖励对比趋势（适合论文使用）"""
    
    # 检查数据
    if 'total_rewards' not in madqn_data or not madqn_data['total_rewards']:
        print("No MADQN reward data found!")
        return
    if 'total_rewards' not in mappo_data or not mappo_data['total_rewards']:
        print("No MAPPO reward data found!")
        return
    
    madqn_rewards = madqn_data['total_rewards']
    mappo_rewards = mappo_data['total_rewards']
    
    # 设置论文级别的字体和样式
    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 1.0,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'serif'
    })
    
    # 创建适合双栏论文的图形尺寸
    plt.figure(figsize=(6, 4.5))
    
    # 使用与test文件一致的配色方案
    madqn_color = '#1ABC9C'  # 青色系 (与test文件中madqn_lp颜色一致)
    mappo_color = '#2E86C1'  # 蓝色系 (与test文件中mappo_lp颜色一致)
    
    # MADQN奖励曲线
    madqn_episodes = range(1, len(madqn_rewards) + 1)
    plt.plot(madqn_episodes, madqn_rewards, alpha=0.3, color=madqn_color, 
             linewidth=0.5, label='MADQN Episode Rewards')
    
    # MADQN移动平均
    if len(madqn_rewards) > 10:
        window_size = min(50, len(madqn_rewards) // 10)
        if window_size > 1:
            smoothed_madqn = []
            for i in range(window_size, len(madqn_rewards)):
                smoothed_madqn.append(np.mean(madqn_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(madqn_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_madqn, 
                    color=madqn_color, linewidth=2.5, label='MADQN Moving Average')
    
    # MAPPO奖励曲线
    mappo_episodes = range(1, len(mappo_rewards) + 1)
    plt.plot(mappo_episodes, mappo_rewards, alpha=0.3, color=mappo_color, 
             linewidth=0.5, label='MAPPO Episode Rewards')
    
    # MAPPO移动平均
    if len(mappo_rewards) > 10:
        window_size = min(50, len(mappo_rewards) // 10)
        if window_size > 1:
            smoothed_mappo = []
            for i in range(window_size, len(mappo_rewards)):
                smoothed_mappo.append(np.mean(mappo_rewards[i-window_size:i]))
            
            smooth_episodes = range(window_size + 1, len(mappo_rewards) + 1)
            plt.plot(smooth_episodes, smoothed_mappo, 
                    color=mappo_color, linewidth=2.5, label='MAPPO Moving Average')
    
    # 论文风格的标题和标签
    plt.title('MADQN vs MAPPO Training Reward Convergence', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Training Episode', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    
    # 与test文件一致的网格样式
    plt.grid(True, alpha=0.3, axis='y')
    
    # 图例放到右上角，与test文件风格一致
    plt.legend(loc='upper right', fontsize=14, framealpha=0.95)
    
    # 调整边距以节省空间
    plt.tight_layout(pad=1.0)
    
    # 保存图形（高质量，适合论文）
    if save_path:
        # 保存为多种格式
        base_name = save_path.rsplit('.', 1)[0]
        
        # PNG格式（高分辨率）
        plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # PDF格式（矢量图，适合论文）
        plt.savefig(f"{base_name}.pdf", bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Comparison plot saved as:")
        print(f"  - {base_name}.png (high-res raster)")
        print(f"  - {base_name}.pdf (vector format)")
    
    plt.show()
    
    # 打印对比统计信息
    print("=" * 60)
    print("MADQN vs MAPPO Training Comparison")
    print("=" * 60)
    print(f"MADQN - Episodes: {len(madqn_rewards)}, Final: {madqn_rewards[-1]:.4f}, Mean: {np.mean(madqn_rewards):.4f}")
    print(f"MAPPO - Episodes: {len(mappo_rewards)}, Final: {mappo_rewards[-1]:.4f}, Mean: {np.mean(mappo_rewards):.4f}")
    print("=" * 60)


def main():
    """主函数：加载数据并绘制奖励趋势"""
    
    # 可用的训练数据文件
    madqn_files = [
        'madqn_training_data_20vehicles_per_exponential.pkl',
        'madqn_training_data_30vehicles_per_exponential.pkl',
    ]
    
    mappo_files = [
        'mappo_training_data_20vehicles.pkl',
        'mappo_training_data_30vehicles.pkl',
    ]
    
    mappo_snr_files = [
        'mappo-snr/mappo_snr_training_data_20vehicles.pkl',
        'mappo_snr_training_data_20vehicles.pkl',  # 备选路径
    ]
    
    print("Available training data files:")
    
    # 检查MADQN文件
    available_madqn = []
    print("\nMADQN files:")
    for i, filename in enumerate(madqn_files):
        if os.path.exists(filename):
            available_madqn.append(filename)
            print(f"{i+1}. {filename}")
    
    # 检查MAPPO文件  
    available_mappo = []
    print("\nMAPPO files:")
    for i, filename in enumerate(mappo_files):
        if os.path.exists(filename):
            available_mappo.append(filename)
            print(f"{i+1}. {filename}")
    
    # 检查MAPPO-SNR文件
    available_mappo_snr = []
    print("\nMAPPO-SNR files:")
    for i, filename in enumerate(mappo_snr_files):
        if os.path.exists(filename):
            available_mappo_snr.append(filename)
            print(f"{i+1}. {filename}")
    
    # 如果三种数据都存在，绘制三方对比图
    if available_madqn and available_mappo and available_mappo_snr:
        print(f"\nThree-way comparison: {available_madqn[0]} vs {available_mappo[0]} vs {available_mappo_snr[0]}")
        try:
            madqn_data = load_training_data(available_madqn[0])
            mappo_data = load_training_data(available_mappo[0])
            mappo_snr_data = load_training_data(available_mappo_snr[0])
            
            save_path = "mappo_snr_vs_mappo_vs_madqn_comparison_paper_figure.png"
            plot_three_way_comparison(madqn_data, mappo_data, mappo_snr_data, save_path)
            return
        except Exception as e:
            print(f"Error loading three-way comparison data: {e}")
            print("Falling back to two-way comparison...")
    
    # 如果只有两种数据都存在，绘制对比图
    if available_madqn and available_mappo:
        print(f"\nComparing: {available_madqn[0]} vs {available_mappo[0]}")
        try:
            madqn_data = load_training_data(available_madqn[0])
            mappo_data = load_training_data(available_mappo[0])
            
            save_path = "madqn_vs_mappo_comparison_paper_figure.png"
            plot_comparison_reward_trends(madqn_data, mappo_data, save_path)
            return
        except Exception as e:
            print(f"Error loading comparison data: {e}")
    
    # 如果只有一种数据，绘制单独的图
    if available_mappo_snr:
        selected_file = available_mappo_snr[0]
        print(f"\nUsing MAPPO-SNR: {selected_file}")
    elif available_madqn:
        selected_file = available_madqn[0]
        print(f"\nUsing MADQN: {selected_file}")
    elif available_mappo:
        selected_file = available_mappo[0]
        print(f"\nUsing MAPPO: {selected_file}")
    else:
        print("No training data files found!")
        print("Expected MADQN files:", madqn_files)
        print("Expected MAPPO files:", mappo_files)
        print("Expected MAPPO-SNR files:", mappo_snr_files)
        return
    
    try:
        # 加载数据
        data = load_training_data(selected_file)
        
        # 显示数据信息
        print(f"Data keys: {list(data.keys())}")
        if 'training_config' in data:
            config = data['training_config']
            print(f"Training config: {config}")
        
        # 生成保存路径（论文格式）
        base_name = selected_file.replace('.pkl', '')
        save_path = f"{base_name}_paper_figure.png"
        
        # 根据文件名自动检测方法类型
        if 'mappo_snr' in selected_file.lower() or 'snr' in selected_file.lower():
            method_type = 'MAPPO-SNR'
        elif 'mappo' in selected_file.lower():
            method_type = 'MAPPO'
        else:
            method_type = 'MADQN'
        
        # 绘制奖励趋势
        plot_reward_trends(data, save_path, method_type)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
