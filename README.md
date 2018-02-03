# grid_walker

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
3 February 2018<br />
git branch=master


### Summary:
The grid_walker demo uses Q-learning to teach a neural net AI
how to navigate an agent as it steps across a square 6x6 grid towards
its goal while avoiding obstacles and hazards.

### Setup:

I execute grid_walker on my Mac laptop, where I installed Anaconda python 2.7 plus
additional libraries:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    chmod +x ./Miniconda2-latest-MacOSX-x86_64.sh
    ./Miniconda2-latest-MacOSX-x86_64.sh -b -p ~/miniconda2
    ~/miniconda2/bin/conda install -y seaborn
    ~/miniconda2/bin/conda install -y jupyter
    ~/miniconda2/bin/conda install -y keras

### Execute:

Start Jupyter notebook via

    jupyter notebook

and load the grid_walker.ipynb notebook

