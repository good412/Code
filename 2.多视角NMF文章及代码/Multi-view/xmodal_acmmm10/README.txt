This demo executes cross-modal retrieval experiments on 
a novel dataset of Wikipedia pages.

It replicates the experiments of the following paper:

@CONFERENCE{rasiwasia2010new,
  author = {Rasiwasia, N. and Costa Pereira, J. and Coviello, E. 
		and Doyle, G. and Lanckriet, G. and Levy, R. 
		and Vasconcelos, N.},
  title = {{A New Approach to Cross-Modal Multimedia Retrieval}},
  booktitle = {ACM International Conference on Multimedia},
  pages={251--260},
  year = {2010}
}

------------------------------------------------------------
 NOTES
------------------------------------------------------------
This code uses some 3rd party applications that you 
replace by similar software packages. 
 - liblinear software package [1]
Matlab's functions are compiled for 64 and 32 bit 
linux platforms. Compile on your system if you have 
a different architecture, or simply change the lookup 
path on matlab.


------------------------------------------------------------
 USAGE
------------------------------------------------------------
Extract the zip file, open a matlab session and 
run any of the example scritps:
 - correlation matching 'demo_CM.m'
 - semantic matching 'demo_SM.m'
 - semantic correlation matching 'demo_SCM.m'


Please send any comments, questions or suggestions to 
 Jose Costa Pereira (josecp@ucsd.edu)



------------------------------------------------------------
 REFERENCES
------------------------------------------------------------
[1]  LIBLINEAR: A Library for Large Linear Classification,
	Rong-En Fan and Kai-Wei Chang and Cho-Jui Hsieh and
		Xiang-Rui Wang and Chih-Jen Lin,
	Journal of Machine Learning Research,
	vol. 9, pp. 1871--1874, 2008

