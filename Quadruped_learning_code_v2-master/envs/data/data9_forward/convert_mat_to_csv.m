clear all
load('jumpingFull_A1_1ms_h20_d50.mat')
load('jumpingFull_A1_1ms_h20_d50_cartesian.mat')
csvwrite('jumpingFull_A1_1ms_h20_d50.csv',data1)
csvwrite('jumpingFull_A1_1ms_h20_d50_cartesian.csv',data2)
clear