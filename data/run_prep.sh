
eval "$(conda shell.bash hook)" #need to run it before running conda activate, it initializes conda
conda activate torch #open the desired environment

python data_preparation_full.py  # Mention the name of your python code file
