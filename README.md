cd Quadruped_learning_code_v2-master
python3 setup.py install --user 
(prepare the MPC environment)

python complex_loco/complex_loco/standing_three.py --task=a1
(run the code)

MPC code is in complex_loco/complex_loco/new_mpc_implementation/
MPClocomotion.py and quadruped.py may need some modification when applied to real robot.