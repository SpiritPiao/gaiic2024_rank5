from PIL import Image
import os

directory = "val_rgb"  # 替换为需要操作的文件  
save_dir = "val_rgb2grey" # 替换为保存的目录路径  

os.makedirs(save_dir, exist_ok=True)

files = []  
for root, dirs, file_names in os.walk(directory):  
    for file_name in file_names:  
        # if extension is None or file_name.lower().endswith(extension.lower()):  
        files.append(os.path.join(root, file_name)) 
        
        image = Image.open(os.path.join(root, file_name))
        rgb_image = Image.new("L", image.size)
        rgb_image.paste(image)
        rgb_image.save(os.path.join(save_dir, file_name))


for file_path in files:  
    print(file_path)