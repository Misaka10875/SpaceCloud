import os
import re
import numpy as np

def check_flat_files_generic(start_dir="."):
    """
    递归扫描指定目录，查找所有不在 'Annotations' 目录下的 .txt 文件。
    如果文件名中不包含 'alignmentAngle'，则进行平面检测。
    如果发现文件内容对应的物体为平面（X、Y 或 Z 轴的极差接近于零），则输出其路径。
    """
    
    MIN_COLS = 3       # 数据文件所需的最少列数 (X, Y, Z)
    TOLERANCE = 0.4   # 浮点数比较阈值：极差小于此值视为平面
    MIN_POINTS = 2     # 至少需要两个点才能计算极差

    flat_objects_found = []
    checked_files_count = 0
    
    # 匹配所有 .txt 结尾的文件
    FILE_PATTERN = re.compile(r".+\.txt$", re.IGNORECASE) 
    
    # 排除关键字
    EXCLUDE_KEYWORD = "alignmentAngle"

    print(f"�� 开始扫描目录: {os.path.abspath(start_dir)}")
    print(f"�� 检测平面极差容忍度: {TOLERANCE}")
    print(f"�� 排除包含 '{EXCLUDE_KEYWORD}' 的文件。\n")

    # 遍历起始目录及其所有子目录
    for root, dirs, files in os.walk(start_dir):
        # 排除所有名为 'Annotations' 的目录及其内容 (不区分大小写)
        if os.path.basename(root).lower() == "annotations":
            continue
        
        for filename in files:
            full_path = os.path.join(root, filename)
            
            # 1. 检查是否为目标文件类型 (.txt)
            if not FILE_PATTERN.match(filename):
                continue
                
            # 2. 检查文件名是否含有排除关键字 (不区分大小写)
            if EXCLUDE_KEYWORD.lower() in filename.lower():
                continue

            checked_files_count += 1
            
            try:
                # 尝试读取文件
                data = np.loadtxt(full_path)
                
                # 检查文件是否为空或格式不正确
                if data.ndim == 0:
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1) # 只有一行数据时转换为 (1, N)
                    
                if data.shape[1] < MIN_COLS:
                    # 列数不足，跳过
                    continue
                    
                coords = data[:, 0:3] 
                
                # 检查点数是否足够
                if coords.shape[0] < MIN_POINTS:
                    # 点数不足，跳过平面检查
                    continue
                    
                # 计算每个轴的极差 (Max - Min)
                ranges = np.ptp(coords, axis=0) 
                
                # 检查是否为平面
                if ranges[0] < TOLERANCE:
                    flat_objects_found.append((full_path, 'X-Flat'))
                elif ranges[1] < TOLERANCE:
                    flat_objects_found.append((full_path, 'Y-Flat'))
                elif ranges[2] < TOLERANCE:
                    flat_objects_found.append((full_path, 'Z-Flat'))
                        
            except Exception as e:
                # 捕获文件读取或处理异常
                print(f"[ERROR] 处理文件 {full_path} 时发生错误: {e}")
                # 继续处理下一个文件

    print("\n" + "="*70)
    print(f"✅ 平面物体检测完成。")
    print(f"�� 共检查了 {checked_files_count} 个目标文件。")
    print("="*70)

    if flat_objects_found:
        print("�� 发现以下平面物体，需要手动检查：")
        for path, reason in flat_objects_found:
            print(f"[{reason}] {path}")
    else:
        print("�� 未发现任何平面物体。")

if __name__ == "__main__":
    # 确保安装了 numpy: pip install numpy
    # 从当前目录开始扫描
    check_flat_files_generic(start_dir=".")
