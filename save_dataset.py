# save_dataset.py
import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO-2017 dataset from FiftyOne's zoo
dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=1000)

# Set the dataset name
dataset.name = "coco-2017-train-1000"

# Save the dataset
dataset.save()

# Ensure the dataset is saved persistently
dataset.persistent = True
dataset.save()

# Verify the dataset is saved correctly
loaded_dataset = fo.load_dataset("coco-2017-train-1000")
print("Dataset loaded successfully:", loaded_dataset)

# Ensure the dataset is listed
datasets = fo.list_datasets()
print("Available datasets:", datasets)


