# grid_walker

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
3 February 2018<br />
git branch=master


### Summary:
This grid_walker demo uses Q-learning to teach a neural net AI how to navigate an agent
#about a 6x6 grid, guiding it towards a goal while avoiding obstacles and hazards.

This version of grid_walker was adapted from a blog post by Outlace,
http://outlace.com/rlpart3.html . Outlace's original code was a sub-optimal mess,
while I like to think this version is less so. However his summary of Q-learning
is excellent and worth a read.

### Setup:

Clone this repo:

    git clone https://github.com/joehahn/grid_walker.git
    cd grid_walker

Note that I am executing grid_walker on a Mac laptop where I've installed
Anaconda python 2.7 plus a few other needed libraries via:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    chmod +x ./Miniconda2-latest-MacOSX-x86_64.sh
    ./Miniconda2-latest-MacOSX-x86_64.sh -b -p ~/miniconda2
    ~/miniconda2/bin/conda install -y seaborn
    ~/miniconda2/bin/conda install -y jupyter
    ~/miniconda2/bin/conda install -y keras

### Execute:

Start Jupyter notebook via

    jupyter notebook

and load the grid_walker.ipynb notebook > Run.

### 

