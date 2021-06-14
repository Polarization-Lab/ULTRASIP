Hi so you are the chosen ULTRASIP data collector...hmmmmm
Okay I think you have what it takes, so let us get started. 

1. Run initializeUV.m
	This program calls CameraConnect and espConnect or ELL14Connect depending on the
	motor being used. Note that if any of their settings need to be changd you will
	have to do it within those scripts. 

2. Run UVsamplerun_automated.m
	This is a fully automated way to take measurements, just make sure the degrees are
	set correctly usually 0,45,90,135. Each time it will ask you to input the altitude
	and azimuth angle for each measurement. Under user notes please put your name 
	and general sky conditions. Output is automatically saved to h5 files named with 
	the time, date and input angles. 

Any problems feel free to contact me,  Clarissa DeLeon. 

