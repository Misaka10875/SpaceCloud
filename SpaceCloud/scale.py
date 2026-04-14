import os
import numpy as np
from scipy.spatial import ConvexHull

def calculate_point_cloud_volume(points):
    """
    计算点云的凸包体积。
    
    Args:
        points (np.array): Nx3的Numpy数组，包含点云的XYZ坐标。
        
    Returns:
        float: 点云的凸包体积。如果点数少于4个，则返回0。
    """
    if points.shape[0] < 4:
        return 0.0
    
    try:
        # 计算凸包
        hull = ConvexHull(points)
        # 返回凸包的体积
        return hull.volume
    except Exception as e:
        print(f"在计算凸包体积时发生错误: {e}")
        return 0.0

def process_point_cloud_files(directory, scale_factor):
    """
    遍历目录及其子目录，处理所有点云文件。
    
    Args:
        directory (str): 待处理的根目录。
        scale_factor (float): 缩放因子。
    """
    print(f"开始处理目录: {directory}")
    print(f"使用的缩放因子为: {scale_factor}")
    print("-" * 50)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"正在处理文件: {file_path}")
                
                try:
                    # 1. 读取点云数据
                    data = np.loadtxt(file_path)
                    
                    # 2. 提取XYZ坐标（前3列）
                    xyz_points = data[:, :3]
                    
                    # 3. 计算缩放前的体积
                    volume_before = calculate_point_cloud_volume(xyz_points)
                    print(f"  > 缩放前的体积: {volume_before:.6f}")
                    
                    # 4. 对XYZ坐标进行缩放
                    scaled_xyz_points = xyz_points / scale_factor
                    
                    # 5. 计算缩放后的体积
                    volume_after = calculate_point_cloud_volume(scaled_xyz_points)
                    print(f"  > 缩放后的体积: {volume_after:.6f}")

                    # 6. 将缩放后的XYZ与原始的RGB数据合并
                    # 确保原始数据有RGB（6列）
                    if data.shape[1] > 3:
                        rgb_data = data[:, 3:]
                        scaled_data = np.hstack((scaled_xyz_points, rgb_data))
                    else:
                        scaled_data = scaled_xyz_points

                    # 7. (可选) 将缩放后的点云保存到新文件
                    # 如果需要保存，可以取消下面几行的注释
                    output_path = file_path.replace('.txt', '_scaled.txt')
                    np.savetxt(output_path, scaled_data, fmt='%.6f')
                    print(f"  > 缩放后的点云已保存至: {output_path}")

                    print("-" * 50)
                
                except Exception as e:
                    print(f"  > 无法处理文件 {file_path}: {e}")
                    print("-" * 50)

# --- 脚本执行入口 ---
if __name__ == "__main__":
    # 定义你的目录路径和缩放因子
    # 例如，如果想将体积缩小到原来的1/10000，那么缩放因子s = 立方根(1/10000)
    # 我们这里使用的脚本是将每个坐标除以scale_factor，所以scale_factor应该大于1
    # 比如我们想把坐标缩小到原来的1/10，那scale_factor就设为10
    # 这样体积就会缩小到原来的1/1000
    root_directory = './'  # 替换为你的点云文件所在的目录
    scale_factor = 1000.0       # 缩放因子。例如，如果你希望x,y,z都除以10

    # 检查目录是否存在
    if not os.path.isdir(root_directory):
        print(f"错误: 目录 '{root_directory}' 不存在。请创建该目录并在其中放入点云文件。")
    else:
        process_point_cloud_files(root_directory, scale_factor)
