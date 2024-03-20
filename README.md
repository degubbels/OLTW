Implementation of an On-line Time Warping algorithm for music alignment.
Developed as part of the opera.guru research project at the Cooperative Systems Research Group, Faculty of Computer Science, University of Vienna.

Code by Dennis Gubbels.

See paper:
O. Hödl, D. Gubbels, O. Shabelnyk and P. Reichl, "Improving a real-time music alignment algorithm for opera performances," 2023 4th International Symposium on the Internet of Sounds, Pisa, Italy, 2023, pp. 1-6, doi: 10.1109/IEEECONF59510.2023.10335462.\
https://eprints.cs.univie.ac.at/7987/1/FINAL%20IEEE%20Improving_a_real-time_music_alignment_algorithm_for_opera_performances.pdf


Please cite/reference Dennis Gubbels, the Cooperative Systems Research Group, Faculty of Computer Science, University of Vienna, University of Vienna and the aforementioned paper in any work that uses the algorithm.

The alignment system is implemented in Python.
The GUI program can be started from OPAQ.py (python ./OPAQ.py).
The system can be used to control some outside service using the Connector class.

The system can of course be further adapted and run without the GUI for more control.
The example.ipynb Jupyter Notebook contains an example for this.
It is also possible to simulate a performance using a second audio file using AudioStream.simulate()

The program uses Librosa for calculating MFCC Features:
McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

This work is licensed under CC BY-NC 4.0 
