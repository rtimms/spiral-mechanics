# A mechanical model for reinforced, expanding spirally-wound layered materials

Code for the paper "A mechanical model for reinforced, expanding spirally-wound layered materials", R. Timms, S. Psaltis, C.P. Please and S.J. Chapman, submitted to Journal of the Mechanics and Physics of Solids, 2022. 

Click [here](https://www.google.com/url?q=https%3A%2F%2Fpapers.ssrn.com%2Fsol3%2Fpapers.cfm%3Fabstract_id%3D4203825&sa=D&sntz=1&usg=AOvVaw1AQJ4_mbEhKUluZoca4KTS) to read a preprint.

To run the files follow the steps below
```bash
# Clone the repository
git clone https://github.com/rtimms/spiral-mechanics/tree/single-wound

# Create a virtual environment in the repository directory
cd spiral-mechanics
virtualenv env

# Activate the virtual environment and upgrade pip
source .venv/bin/activate
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt
```

Note: The scripts all assume you are running from the root 'spiral-mechanics' folder.
