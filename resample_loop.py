import os
import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

def resample_nifti(input_path, output_path, target_num_slices, is_label=False):
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    current_num_slices = data.shape[-1]
    x_original = np.linspace(0, 1, current_num_slices)
    x_resampled = np.linspace(0, 1, target_num_slices)
    if is_label:
      
        interpolation_func = interp1d(x_original, data, kind='nearest', axis=-1, fill_value="extrapolate")
    else:
    
        interpolation_func = interp1d(x_original, data, kind='linear', axis=-1, fill_value="extrapolate")
    resampled_data = interpolation_func(x_resampled)
    new_nifti_img = nib.Nifti1Image(resampled_data, nifti_img.affine)
    nib.save(new_nifti_img, output_path)
def resample_multiple_niftis(input_image_dir, output_image_dir, input_label_dir, output_label_dir, target_num_slices):

    input_image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.nii.gz')]
    input_label_files = [f for f in os.listdir(input_label_dir) if f.endswith('.nii.gz')]


    assert len(input_image_files) == len(input_label_files), "Mismatch in number of image and label files"

    for image_file, label_file in zip(input_image_files, input_label_files):
        input_image_path = os.path.join(input_image_dir, image_file)
        output_image_path = os.path.join(output_image_dir, image_file)
        input_label_path = os.path.join(input_label_dir, label_file)
        output_label_path = os.path.join(output_label_dir, label_file)
        
        resample_nifti(input_image_path, output_image_path, target_num_slices, is_label=False)
        resample_nifti(input_label_path, output_label_path, target_num_slices, is_label=True)

input_image_dir = '/mnt/myhdd/Abdominal_1k/zerohataune/images/'
output_image_dir = '/home/kabin/Data_Train_Test/TestVolumes/'
input_label_dir = '/mnt/myhdd/Abdominal_1k/zerohataune/labels/'
output_label_dir = '/home/kabin/Data_Train_Test/TestSegmentation/'
target_num_slices = 50  # Adjust this value to the desired number of slices
resample_multiple_niftis(input_image_dir, output_image_dir, input_label_dir, output_label_dir, target_num_slices)
