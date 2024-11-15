import os
import random
import pickle

# Define the path to the dataset folder
dataset_folder = os.path.join(os.path.dirname(os.getcwd()), 'Crash_dataset')

# Get the list of case IDs (subfolder names)
case_ids = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]

# Set a random seed for reproducibility
random.seed(42)

# Shuffle the case IDs to ensure randomness
random.shuffle(case_ids)

# Split the case IDs into training, validation, and testing sets
# validation_set = case_ids[:15]
# testing_set = case_ids[15:]

# Save the different set lists as .pkl files
# with open('validation_set.pkl', 'wb') as f:
#     pickle.dump(validation_set, f)
# 
# with open('testing_set.pkl', 'wb') as f:
#     pickle.dump(testing_set, f)
#
# print("Validation and testing sets saved successfully.")

full_set = case_ids
with open('full_set.pkl', 'wb') as f:
    pickle.dump(full_set, f)