import os
import argparse
import glob
import shutil
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_realsr", type=str)
parser.add_argument("--path_to_realsr_reordered", type=str)
parser.add_argument("--factor", type=int)
opt = parser.parse_args()

path_to_realsr = opt.path_to_realsr
path_to_realsr_reordered = opt.path_to_realsr_reordered

factor = opt.factor

path_to_realsr_reordered = opt.path_to_realsr_reordered
os.makedirs(path_to_realsr_reordered, exist_ok=True)

path_to_realsr_reordered_hr = os.path.join(path_to_realsr_reordered, f"HR_x{factor}")
os.makedirs(path_to_realsr_reordered_hr, exist_ok=True)
path_to_realsr_reordered_lr = os.path.join(path_to_realsr_reordered, f"LR_x{factor}")
os.makedirs(path_to_realsr_reordered_lr, exist_ok=True)


camera_paths = sorted(glob.glob(os.path.join(path_to_realsr, "*")))

for camera_path in camera_paths:
    camera_name = os.path.basename(camera_path)
    print(f"Camera {camera_name}")
    path_to_test_camera = os.path.join(camera_path, "Test")
    path_to_test_camera_factor = os.path.join(path_to_test_camera, str(factor))
    HR_data = sorted(glob.glob(os.path.join(path_to_test_camera_factor, f"{camera_name}_*_HR.png")))
    HR_data_basenames = [os.path.basename(path) for path in HR_data]
    LR_data = sorted(glob.glob(os.path.join(path_to_test_camera_factor, f"{camera_name}_*_LR{factor}.png")))
    LR_data_basenames = [os.path.basename(path) for path in LR_data]

    print(f"Reordering RealSR HR data for factor {factor}")
    for hr_data_basename in tqdm.tqdm(HR_data_basenames):
        path_to_hr_image = os.path.join(path_to_test_camera_factor, hr_data_basename)
        index = hr_data_basename.split("_")[1]
        hr_data_basename_new = f"{camera_name}_{index}.png"
        new_path_image = os.path.join(path_to_realsr_reordered_hr, hr_data_basename_new)
        shutil.copy(path_to_hr_image, new_path_image)

    print(f"Reordering RealSR LR data for factor {factor}")
    for lr_data_basename in tqdm.tqdm(LR_data_basenames):
        path_to_lr_image = os.path.join(path_to_test_camera_factor, lr_data_basename)
        index = lr_data_basename.split("_")[1]
        lr_data_basename_new = f"{camera_name}_{index}.png"
        new_path_image = os.path.join(path_to_realsr_reordered_lr, lr_data_basename_new)
        shutil.copy(path_to_lr_image, new_path_image)
