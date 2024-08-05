#!/bin/bash

# Define the range of folds and the list of model names
model_names=('Unet' 'UnetPlusPlus' 'FPN' 'DeepLabV3')
yaml_file="configs/unet_family.yaml"  # Path to your YAML file

# Function to update the YAML file
update_yaml() {
    model_name=$1
    python - <<EOF
import yaml

yaml_file = '$yaml_file'
model_name = '$model_name'

# Load YAML file
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Update values
config['model_name'] = model_name

# Save updated YAML file
with open(yaml_file, 'w') as file:
    yaml.safe_dump(config, file)
EOF
}

runner_file="training/trainer.py"  # Path to your Python script

# Iterate over each model name
for model_name in "${model_names[@]}"; do
    echo "Updating YAML and running $runner_file with model_name=$model_name"

    # Update the YAML file with current model name
    update_yaml $model_name

    # Run the Python script
    python $runner_file &

    # Wait for the previous run to complete
    wait

    echo "Completed model_name = $model_name"
done

echo "All combinations have been processed."
