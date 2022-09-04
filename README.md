# CricketLytics 🏏
[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Autopilot/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

Cricket analytics for humans

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

## Description 🎾
Cricket is a bat-and-ball game played between two teams of eleven players on a field at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps. 

The batting side scores runs by striking the ball bowled at the wicket with the bat (and running between the wickets), while the bowling and fielding side tries to prevent this (by preventing the ball from leaving the field, and getting the ball to either wicket) and dismiss each batter (so they are "out"). 

## Python  Implementation 👨‍🔬

**Supported Trainining**

-  Batting (Front View)
-  Batting (Side View)
-  Bowling (Front View)

**Source**
- '0' for webcam
- Any other source for a prerecorded video

If you face any problem, kindly raise an issue

## Setup 🖥️

1) First, record the cricket training session you want to perform analytics on; or you can setup your webcam so that it can stream your cricket session in runtime.
2) Select the type of exercise you want to perform. (Look above for the supported exercises)
3) Run the `Cricket.py` file with your current configuration

## Execution 🐉

```
python Cricket.py --option <batting or bowling> --hand <left or right> --view <front or side>
```

`python Cricket.py --option batting --hand right --view front`


## Results 📊


<div align="center">
<img src="https://github.com/akshaybahadur21/BLOB/blob/master/crick.gif" width=600>
</div>

## References 🔱
 
 -  Ivan Grishchenko and Valentin Bazarevsky, Research Engineers, Google Research. [Mediapipe by Google](https://github.com/google/mediapipe)
