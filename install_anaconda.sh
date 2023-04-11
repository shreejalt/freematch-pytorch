# Taken from https://github.com/ericmjl/dotfiles/blob/master/install_functions.sh

function install_anaconda {
  bash anaconda.sh -b -p $HOME/anaconda
  rm anaconda.sh
  export PATH=$HOME/anaconda/bin:$PATH

  # Install basic data science stack into default environment
  conda install --yes pandas scipy numpy matplotlib seaborn jupyter ipykernel nodejs

  jupyter notebook --generate-config
  # We are done at this point, move on.
  echo "anaconda successfully installed. moving on..."
}

# Taken from https://github.com/ericmjl/dotfiles/blob/master/install.sh#L108

wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O anaconda.sh
install_anaconda