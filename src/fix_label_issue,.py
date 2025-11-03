import os
from pathlib import Path
import tqdm


# MAKE BY GEMINI


def fix_label_class_id_format(labels_dir: str):
    """
    遍历指定目录下的所有 .txt 标签文件，将每行标签的第一个字段
    （class_id，如果它是浮点数格式，如 '2.0'）修改为整数格式（如 '2'）。

    Args:
        labels_dir: 包含 YOLO 标签文件（.txt）的目录路径。
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.is_dir():
        print(f"❌ 错误：指定的路径不是一个有效的目录：{labels_dir}")
        return

    # 查找所有 .txt 文件
    label_files = list(labels_path.glob('**/*.txt'))
    
    if not label_files:
        print(f"⚠️ 警告：在目录 {labels_dir} 中未找到任何 .txt 标签文件。")
        return

    print(f"📂 开始处理 {len(label_files)} 个标签文件...")

    for file_path in tqdm.tqdm(label_files, desc="正在处理标签文件"):
        try:
            # 1. 读取文件所有内容
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            modified = False

            # 2. 逐行处理
            for line in lines:
                parts = line.strip().split()
                
                # 检查是否是标准的YOLO格式行
                if len(parts) >= 5:
                    first_field = parts[0]
                    
                    # 尝试将第一个字段转换为浮点数，如果成功，则进一步处理
                    try:
                        # 如果第一个字段是 '2.0' 这样的浮点数
                        float_value = float(first_field)
                        
                        # 检查它是否应该被转换（即它的小数部分是 .0）
                        # 示例：float_value == int(float_value) 意味着 2.0 == 2
                        if float_value == int(float_value):
                            # 将 '2.0' 转换为 '2'
                            parts[0] = str(int(float_value))
                            modified = True
                        
                    except ValueError:
                        # 如果无法转换为浮点数（它本身就是正确的整数或其它字符串），则忽略
                        pass
                
                # 3. 重新组合行并保留原始的换行符
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)

            # 4. 如果文件被修改过，则覆盖写入
            if modified:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                # print(f"✅ 已修改: {file_path.name}") # 可选：打印被修改的文件
            
        except Exception as e:
            print(f"❌ 处理文件 {file_path.name} 时发生错误: {e}")

    print("🎉 所有标签文件处理完成。")


def main():
    # 指定包含 YOLO 标签文件的目录
    labels_directory = os.path.join(Path(__file__).parent.parent.resolve(), 'transformed_dataset', 'labels', 'train')
    
    fix_label_class_id_format(labels_directory)


if __name__ == "__main__":
    main()