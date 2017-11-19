make
time ./word2vec -train words -output vectors.bin -cbow 1 -size 200 -window 5 -negative 100 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 30
./distance vectors.bin
