import kagglehub

# Download latest version
path = kagglehub.dataset_download("divyansh22/intel-berkeley-research-lab-sensor-data")

print("Path to dataset files:", path)