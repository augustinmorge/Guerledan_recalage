# Recalage à l'aide de bathymétrie pour un AUV
***
We want to recalibrate the state of the AUV using the bathymetry and particle filter.

## Folders
### Documentation
The bibliography.
* DVL
* Particle Filter
* Subject

### Resources
Existing programs on our subject to run :
* DVL
* Particle Filter
* INS

### Simulation
Run a simulation using or not the data that we got for the previous Guerledan.

## Useful links
Manual of the different sensors used
[SOURCE](https://moodle.ensta-bretagne.fr/course/view.php?id=863)

Youtube link for basic explanation of the particulate filter
[SOURCE](https://www.youtube.com/watch?v=NrzmH_yerBU&ab_channel=MATLAB)

## TODO
* Decipher the binary data from the DVL
* Interpolate the DTM
* Recover single-beam data

***
## Sensors available
* DVL: Bottom speed/distance slant (no current speed)
* INS: Gyroscope/Accelerometer -> Speed and position (see how to do)
* GNSS: RTK (to be processed)
* MBES multibeam: bathymetry (equi-angle?)
* SVP: Hull velocity meter
