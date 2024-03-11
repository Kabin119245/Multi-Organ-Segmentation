import SimpleITK as sitk

# Load the .nii file
image = sitk.ReadImage('/mnt/myhdd/Data_Train_Test/TrainSegmentation/Case8.nii.gz')

# Get the unique labels present in the image
unique_labels = sitk.GetArrayFromImage(image)
unique_labels = set(unique_labels.flatten())  # Flatten the array and convert to a set to get unique labels

# Remove background label if present (optional)
if 0 in unique_labels:
    unique_labels.remove(0)

# Count the number of unique labels
num_labels = len(unique_labels)

print("Number of unique labels:", num_labels)
