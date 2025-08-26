# DLA Toolkit (DLAT)

DLA detection in quasar spectra via template fitting

# Installation

This repository has desispec dependencies. The code is straightforward to run for NERSC users with the main DESI environment loaded.

Clone the repository, add the "py" directory to your $PYTHONPATH, and the "bin" directory to your $PATH to install the DLA Toolkit.


# Example

Sample commands to run the DLA Toolkit on the main-dark survey from a given DESI redux on a NERSC interactive node. 
Set $QSOCAT, $REDUX, and $OUTDIR variables accordingly, then
```
source /global/common/software/desi/desi_environment.sh main
desi_dlatoolkit.py -q $QSOCAT -r $REDUX -p dark -s main \
       	--balmask -o $OUTDIR --outfile dlacat-$REDUX-main-dark.fits 
``` 

The default nproc=64 was optimized for the kibo reduction. You can manually set this with the `--nproc` argument. 
If not submitting a batch job or running on an interactive node, specify `--nproc 1` in command.

The commands for running the DLA Toolkit on a mock data set are similar.
Set the $MOCKDIR, $QSOCAT, $REDUX, and $OUTDIR variables accordingly, then:
```
source /global/common/software/desi/desi_environment.sh main
desi_dlatoolkit.py -q $QSOCAT -r $REDUX --mock --mockdir $MOCKDIR \
        --balmask -o $OUTDIR --outfile dlacat-$REDUX-mock-main-dark.fits
```


