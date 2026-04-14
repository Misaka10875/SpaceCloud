import os
import re

def find_unique_labels(start_dir="."):
    """
    递归扫描指定目录下所有名为 'Annotations' 的文件夹，提取并输出不重复的标签 (x)。
    标签格式假定为 'x_n.txt'，其中 x 和 n 都可以包含字母、数字或下划线。
    """
    
    unique_labels = set()
    annotation_folders_found = 0
    files_processed = 0

    # 正则表达式说明:
    # ^: 匹配字符串开始
    # (.+?): 捕获组 1 (标签 x)。.+? 匹配至少一个任意字符，非贪婪模式
    # _: 匹配下划线
    # \d+: 匹配至少一个数字 (n)
    # \.txt$: 匹配 .txt 结尾
    # 注意: 如果 n 包含非数字，需要将 \d+ 替换为 [a-zA-Z0-9_-]+
    # 假设 'n' 至少包含数字，'x' 包含字母数字下划线
    LABEL_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+?)_\d+\.txt$", re.IGNORECASE)

    # 遍历起始目录及其所有子目录
    for root, dirs, files in os.walk(start_dir):
        # 检查当前目录是否为目标 Annotations 文件夹
        if os.path.basename(root).lower() == "annotations":
            annotation_folders_found += 1
            
            print(f"-> 扫描文件夹: {root}")
            
            for filename in files:
                files_processed += 1
                
                # 尝试匹配文件名
                match = LABEL_PATTERN.match(filename)
                
                if match:
                    # 提取捕获组 1，即标签 x
                    label_x = match.group(1)
                    unique_labels.add(label_x)

    print("\n" + "="*50)
    print(f"✅ 扫描完成。")
    print(f"�� 找到 {annotation_folders_found} 个 Annotations 文件夹。")
    print(f"�� 处理了 {files_processed} 个文件。")
    print("="*50)

    if unique_labels:
        print("�� 发现的不重复标签 (x) 如下：")
        # 按字母顺序输出标签
        for label in sorted(list(unique_labels)):
            print(f"- {label}")
    else:
        print("⚠️ 未发现符合 'x_n.txt' 命名格式的文件，或者所有 Annotations 文件夹为空。")


if __name__ == "__main__":
    # 从当前目录开始扫描
    # 如果需要在其他目录执行，请修改 start_dir 参数
    find_unique_labels(start_dir=".")
