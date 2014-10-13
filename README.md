Matrix_Factorization
====================
This repository includes has a number of matrix factorization techniques including Nystrom and OASIS. The codes are written in c++ using MPI for distributed computing.

example usage:

To complie:

mpic++ -o main.out /src/oasiskernel.cpp

To run:

mpiexec -n 8 main.out /data/matrix.txt 2000 2 100 10 .001 // 8 is th number of parallel nodes
