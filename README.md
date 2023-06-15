# Landscape Booth - LNDW 2023

Repository for GFZ section 4.7's contribution to the Long Night of Sciences LNDW 2023.

https://www.langenachtderwissenschaften.de/

## Booth variants

- ``booth1.ipynb``: the ``Record`` button triggers continuous update of a vertical uplift field computed from the webcam stream (with face detection). When the same button is clicked again, it freezes the uplift field then resets it to zero (relaxation) after a few time steps.

- ``booth2.ipynb``: the ``Take picture`` button only takes a single frame from the webcam and convert it into an uplift field (with face detection). That field is applied for a few time steps before being reset to zero (relaxation).

## Installation

1. clone the git repository (you will need your GFZ username and *domain* password):

```
git clone https://git.gfz-potsdam.de/sec47/lndw2023
```

2. cd into the cloned repository:

```
cd lndw2023
```

3. Create a new conda envrionment using `conda` or `mamba`:

```
mamba create --file environment.yml
```

4. Activate the environment:

```
conda activate lndw2023
```

## Run the Booth

You can run one of the booths as a standalone web app using ``voila``. Run the following command from the repository root directory (don't forget to first activate the conda environment if not already activated):

```
voila booth1.ipynb
```

This should automatically open a new tab in your web browser.

Alternatively (for development) you can run jupyter lab:

```
jupyter lab
```

## Usage

### Allow your web browser to access your webcam

When clicking the ``Record`` (booth 1) or ``Take picture`` (booth 2) button for the first time, your browser should ask your permission to access the webcam.

### Save a snapshot to a file

Click on the ``Snapshot to file`` button to take a snapshot picture and save it to a file. A filename must be given in the text input next to the button. The file is saved in the ``snapshots`` sub-folder in the repository's root directory.

## Troubleshooting

The face detection algorithm works best under the following settings:

- a white and uniform background
- good lighting
- only one person facing the webcam
- the person's face centered in front of the webcam

Despite we did our best to make it stable, it is possible that the booth crashes, freezes or does nothing while running. It is also possible that the system load is too high (CPUs, fans, memory). If that's the case, try the following (in order):

- Press again on the ``(Stop) Record`` (booth 1) or ``Take picture`` (booth 2) button on the top right (sometimes the face detection algorithm fails)
- ``Stop / Reset`` and ``Start`` again the simulation
- Restart the Kernel and re-run the notebook (when in JupyterLab)
- Refresh the web-browser tab (Voila or JupyterLab)
