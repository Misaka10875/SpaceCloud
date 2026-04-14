import os
import argparse
import sys

def get_line_count(file_path):
    """
    计算文件的行数。
    """
    try:
        # 使用生成器高效地按行读取文件，不将整个文件加载到内存
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = sum(1 for line in f)
        return line_count
    except Exception as e:
        print(f"警告: 无法读取文件 '{file_path}'，错误: {e}", file=sys.stderr)
        return 0

def find_large_files(directory, max_lines, exclude_keyword):
    """
    递归扫描指定目录，找出内容行数超过指定阈值的.txt文件。

    参数:
    directory (str): 要扫描的起始目录。
    max_lines (int): 文件行数阈值。
    exclude_keyword (str): 要排除的文件名中的关键字。

    返回:
    list: 包含符合条件的文件路径的列表。
    """
    large_files = []
    print(f"正在扫描目录: {os.path.abspath(directory)}")
    print(f"查找行数超过 {max_lines} 行的.txt文件，并排除包含 '{exclude_keyword}' 的文件。")
    print("-" * 50)

    for root, _, files in os.walk(directory):
        for filename in files:
            # 检查文件是否为.txt文件且不包含排除关键字
            if filename.endswith(".txt") and exclude_keyword not in filename:
                file_path = os.path.join(root, filename)
                try:
                    line_count = get_line_count(file_path)
                    if line_count > max_lines:
                        large_files.append((file_path, line_count))
                except Exception as e:
                    print(f"处理文件 '{file_path}' 时发生错误: {e}", file=sys.stderr)

    return large_files

def main():
    parser = argparse.ArgumentParser(description="查找指定目录下行数超过50万的.txt文件。")
    parser.add_argument(
        "directory",
        type=str,
        default=".",
        nargs="?",
        help="要扫描的起始目录（默认为当前目录）。"
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在。")
        return

    # 设定参数
    MAX_LINES = 300000
    EXCLUDE_KEYWORD = "Angle"

    found_files = find_large_files(args.directory, MAX_LINES, EXCLUDE_KEYWORD)

    if found_files:
        print("\n找到以下符合条件的文件:")
        for file_path, line_count in found_files:
            print(f"{file_path} - {line_count} 行")
    else:
        print("\n未找到符合条件的文件。")

if __name__ == "__main__":
    main()
