import os

# Assuming the path to the folder containing unlabeled images
unlabeled_folder = 'good_new'

# List all files in the folder
files = os.listdir(unlabeled_folder)

# Sort the files in the folder (optional, for consistent naming)
files.sort()

# Label and rename the files in the folder
for i, file in enumerate(files):
    # Generate the new filename in the format frame0001.jpg, frame0002.jpg, etc.
    new_filename = f'frame{str(i + 1).zfill(4)}.jpg'
    
    # Full path to the old and new file
    old_path = os.path.join(unlabeled_folder, file)
    new_path = os.path.join(unlabeled_folder, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)
