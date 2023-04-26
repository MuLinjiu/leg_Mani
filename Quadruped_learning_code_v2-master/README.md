# Learning
Cleaned up learning codebase for quadruped robot learning

## Installation

Then, install usc_learning as a package:

`python3 setup.py install --user `
or 
`pip3 install -e . `


## Code structure

- [learning](./usc_learning/learning) training and testing scripts
- [envs](./usc_learning/envs) robot class and task environment for trainning
- [tests] function tests for robot classes and task envrionments

## Status
- PMTG locomotion task with IK
- end-to-end locomotion task with Cartesian control
- end-to-end locomotion task in joint space
- MPC task to learn foothold offset from heuristic
- manipulation task

## TODO
- reimplement MPC
- migrate jumping