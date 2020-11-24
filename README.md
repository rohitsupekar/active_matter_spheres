Repository for:

[Linearly forced fluid flow on a rotating sphere](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/linearly-forced-fluid-flow-on-a-rotating-sphere/1D39336D18A0F7BDEB56E6C082E6F376)

R. Supekar, V. Heinonen, K. J. Burns & J. Dunkel 

Major dependency: 

Dedalus ([install using conda](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html#conda-installation-recommended))

For a standalone run, do the following:

`cd standalone`

`conda activate dedalus`

Specify the run parameters and output folder in `standalone/runningSimulation.py`. 

Run the script:

`mpiexec -n <ncores> python3 runningSimulation.py`

For an example of plotting and making a video from the saved data:

`mpiexec python3 plot_output.py`
