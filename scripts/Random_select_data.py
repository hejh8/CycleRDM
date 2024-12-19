import os
import random
import shutil

def copy_corresponding_images(source_folder1, source_folder2, destination_folder1, destination_folder2, num_images=450):
    # 获取两个源文件夹中的所有图片文件名
    files1 = os.listdir(source_folder1)
    files2 = os.listdir(source_folder2)
    
    # 找到两个文件夹中共同的图片文件名
    common_files = list(set(files1) & set(files2))
    
    # 从共同的图片文件名中随机选择指定数量的图片文件
    selected_files = random.sample(common_files, num_images)
    
    # 将选定的图片文件复制到目标文件夹
    for file in selected_files:
        source_path1 = os.path.join(source_folder1, file)
        source_path2 = os.path.join(source_folder2, file)
        destination_path1 = os.path.join(destination_folder1, file)
        destination_path2 = os.path.join(destination_folder2, file)
        
        shutil.copyfile(source_path1, destination_path1)
        shutil.copyfile(source_path2, destination_path2)
        
        print(f"Copied {file} to {destination_path1} and {destination_path2}")

# 示例用法
source_folder1 = '/home/ubuntu/Image-restoration/CycleRDM/data/deblurry/BSD/blur'
source_folder2 = '/home/ubuntu/Image-restoration/CycleRDM/data/deblurry/BSD/gt'
destination_folder1 = '/home/ubuntu/Image-restoration/CycleRDM/data/deblurry/BSD/1'
destination_folder2 = '/home/ubuntu/Image-restoration/CycleRDM/data/deblurry/BSD/2'

copy_corresponding_images(source_folder1, source_folder2, destination_folder1, destination_folder2, num_images=500)
