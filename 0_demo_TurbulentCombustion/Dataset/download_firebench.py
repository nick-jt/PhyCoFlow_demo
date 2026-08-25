import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastnet/firebench-u10-2")

print("Path to dataset files:", path)