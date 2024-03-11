import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

def resample_nifti(input_path, output_path, target_num_slices, is_label=False):
    # Load the NIfTI file
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()

    # Get the current number of slices
    current_num_slices = data.shape[-1]

    # Create an interpolation function for resampling
    x_original = np.linspace(0, 1, current_num_slices)
    x_resampled = np.linspace(0, 1, target_num_slices)

    if is_label:
        # Use nearest-neighbor interpolation for label data
        interpolation_func = interp1d(x_original, data, kind='nearest', axis=-1, fill_value="extrapolate")
    else:
        # Use linear interpolation for image data
        interpolation_func = interp1d(x_original, data, kind='linear', axis=-1, fill_value="extrapolate")

    # Resample the data using the interpolation function
    resampled_data = interpolation_func(x_resampled)

    # Create a new NIfTI image with resampled data
    new_nifti_img = nib.Nifti1Image(resampled_data, nifti_img.affine)

    # Save the resampled NIfTI file in .nii.gz format
    nib.save(new_nifti_img, output_path)

# Example usage
input_image_path = '/mnt/myhdd/Abdominal_1k/original/imagesTr/Case_00418_0000.nii.gz/'
output_image_path = '/mnt/myhdd/Data_Train_Test/TrainVolumes/Case16.nii.gz/'
input_label_path = '/mnt/myhdd/Abdominal_1k/original/labelsTr/Case_00418.nii.gz/'
output_label_path = '/mnt/myhdd/Data_Train_Test/TrainSegmentation/Case16.nii.gz/'
target_num_slices = 50  # Adjust this value to the desired number of slices

# Resample image data
resample_nifti(input_image_path, output_image_path, target_num_slices, is_label=False)

# Resample label data
resample_nifti(input_label_path, output_label_path, target_num_slices, is_label=True)
