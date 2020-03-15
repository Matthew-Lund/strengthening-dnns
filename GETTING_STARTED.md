# Getting Started

## Clone this repository

Navigate to the folder that you would like to store this repository in and clone the repository.

```
git clone git@github.com:katywarr/strengthening-dnns.git
```
To keep up-to-date with any changes to the repository:

```
cd strengthening-dnns
git pull
```

# Setting up your environment with Anaconda

The Anaconda package contains python and many of the required dependencies for this project.
Instructions for downloading it are [here](https://docs.anaconda.com/anaconda/install/)

In recent versions of Anaconda, it is recommended that you do *not* select the option to add 
the Anaconda bin folder to your PATH during installation but use the Anaconda Prompt.

## Using the Anaconda Prompt 

On windows: Click start and start typing "Anaconda" to get the prompt.


## Create a virtual Python environment (one-time)

From within an Anaconda command prompt, navigate to the `strengthening-dnns` folder and
create a virtual environment using the following command: 

*You only need to do this once.*

```
conda env create -f strengthening-dnns.yml 
```

Whenever you want to use this environment, invoke:

```
conda activate strengthening-dnns
```

Your prompt should now look like something like this:

```
(strengthening-dnns) current_dir>
```

# Running the code


From within an Anaconda Command Prompt, ensure that you are within the correct environment 

```
conda activate strengthening-dnns
```

Navigate to the folder containing the clone of this repository and type:

```
jupyter notebook
```

Here's a good [introduction to Jupyter notebooks](https://jupyter-notebook-beginner-guide.readthedocs.io)



