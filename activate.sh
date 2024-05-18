echo "Jai Shree Ram"
user_name=$(whoami)
path_to_env="/home/${user_name}/.cache/pypoetry/virtualenvs/computer-vision-*"
if [ -d $path_to_env ]; then
  echo "Found the existing environment!!"
else
  echo "Creating a working environment ..."
  poetry install --no-root
  
fi
echo -e "\n\nActivating the working environment..."


source .env
read  -p "Enter wandb preference : " wandb_usr
if [ $wandb_usr == "sarvagya" ]; then
  echo "Setting up wandb..."
  wandb login --relogin "$Sarvagya_WANDB_API"
fi

poetry shell