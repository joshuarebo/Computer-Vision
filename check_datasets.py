# check_datasets.py
import fiftyone as fo

# List all datasets in FiftyOne
datasets = fo.list_datasets()
print("Available datasets after saving:", datasets)
