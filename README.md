[Project In-Porgress]

# Install the venv tool
sudo apt update && sudo apt install python3-venv -y

# Create a virtual environment named '.venv'
python3 -m venv .venv

# Activate the virtual environment 
source .venv/bin/activate

# Install dependencies 
pip install garminconnect curl_cffi 
pip install pandas 
pip install scikit-learn

# Fetch Garmin data via Garmin API 
python fetch-garmin.py

# Process the Garmin data to build the datasets 
python preprocessing.py

# To train models 
python train.py --mode train 

# Leave the virtual environment
deactivate