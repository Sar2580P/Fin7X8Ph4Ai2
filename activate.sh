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
echo "Setting up wandb..."
if [ $wandb_usr == "sarvagya" ]; then
  wandb login --relogin "$Sarvagya_WANDB_API"
else
  wandb login --relogin "$Ishan_WANDB_API"
fi

poetry shell

