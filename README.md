# Recalage à l'aide de bathymétrie pour un AUV
***
The objective of this project is to realign and therefore position an AUV. For this purpose it is assumed that it moves on an already known DTM and that by comparing the values with the background it can estimate its position with the help of a particle filter.

However, we have simplified this problem. Here it will only be a boat. It will move on the plane and so we can compare with the real GPS values.

## Folders
### Documentation
In this folder we have the bibliography used for the project.

* Pathfinder (DVL)
* Filter (Particule Filter)
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

***
## Sensors available
* DVL: Bottom speed/distance
<div style="text-align:center">
<p align="center">
  <img src="https://geo-matching.com/uploads/default/m/i/migrationtaxfjy.jpg" width="750" title="Ragazza Proboat">
</p>
</div>
* INS: Gyroscope/Accelerometer -> Speed and position
* GNSS
* MBES multibeam: bathymetry
* SVP: Hull velocity meter

## Participants
Emails :

[Léo Pham-Van](mailto:leo.pham-van@ensta-bretagne.org) Hydrographe - Prix Nobel

[Laurent Droudun](mailto:laurent.droudun@ensta-bretagne.org) Roboticien - Médaille Fields

[Florian Jerram](mailto:florian.jerram@ensta-bretagne.org) Roboticien - Prix Goncourt

[Augustin MORGE](mailto:augustin.morge@ensta-bretagne.org) Roboticien - Meilleur patissier
