/**The MIT License (MIT)

 * Copyright (c) 2014 Rice University

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Implimentation of Nysrom algorithm for a Gaussian kernel matrix 
 * Nystrom is a random column subset selection method that enables factorizing massive kernel matrices: G = C * invw * C^T
 * [m n] = size of matrix A
 * k = number of columns to be selected from G at random; [n k] = size of matrix C
 * sigma = sigma of the Gaussian kernel exp(-(||ai-aj||_2)^2/sigma)
 * Code written by Azalia Mirhoseini  
 */



#include <mpi.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#define err_countMax 10000 // maximum number of columns chosen for error analysis per core


using namespace std;
using namespace Eigen;




double kernelf(const Ref<const MatrixXd>& a1, const Ref<const MatrixXd>& a2,  double sig)
{
	double temp;
	
	temp = exp(-((a1-a2).norm()*(a1-a2).norm())/sig);
	return temp;
	}   
int main(int argc, char *argv[])
{
	
	    int npes, myrank;
        double t1, t2, tread1, tread2;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        srand(time(0) + myrank);

        if (argc<7)
        {
                if(!myrank)
                        cout << "Please enter path to matrix A, n,  m,  k, s, sigma"<<endl;
				MPI_Finalize();
                return -1;
        }

		
    
	    uint64_t n = atoi(argv[2]);
        uint64_t m = atoi(argv[3]);
        int k = atoi(argv[4]);
        int s = atoi(argv[5]);
        int init_s =s;
        
	   if (s>n)
        {
                if(!myrank)
                        cout << " k (number of randomly selected columns) should be less than n (total number of columns)"<<endl;
				MPI_Finalize();
                return -1;
        }
        
        double sig = atof(argv[6]);
	

		
		uint64_t myn = (n+(npes-1))/npes;
		uint64_t mystartid= myrank*myn;
		MPI_Barrier(MPI_COMM_WORLD);
		int neg = (n-((npes-1)*myn));
		if (neg<0)
        {
                if(!myrank)
                        cout << "Process Terminated: Increase n or Reduce ncpu"<<endl;
				MPI_Finalize();
                return -2;
        }
        
   
		
        if(myrank==npes-1)
        {
                myn = n - mystartid;  
        }
	
		MatrixXd A(m,n);
		MatrixXd C(myn,k);
		MatrixXd R(k,myn);
		MatrixXd w(k,k);
		MatrixXd invw(k,k);
        MatrixXi rand_vec(n,1);
		
		if(!myrank)
			{
			printf( "\033[32;1mStart loading A from file\033[0m\n");
			}

			
			tread1 = MPI_Wtime(); 
				ifstream fr;
				fr.open(argv[1]);
				
				for(uint64_t i=0; i< m; i++)
				
				{
					for(uint64_t j=0; j< n; j++)
					{
						fr>>A(i,j);
						}
					}	
					
					fr.close();
					
					tread2 = MPI_Wtime(); 
					
			
			
			if(!myrank)
			{
			printf( "\033[32;1mDone loading A from file\033[0m\n");
			}
	
			if(!myrank)
			printf( "\033[1;31mMatrix A loading time = %fs\033[0m\n",(tread2 - tread1));
                
                
		
		
		/*create w by randomly selecting #s input indices*/
        t1 = MPI_Wtime();
		uint64_t s_ind[k]; 
		uint64_t range  = n;
		
		if(!myrank)
		{
            for(int i=0; i<n; i++ )
            {
                rand_vec(i,0) = i;
            }
            for(int i=n-1; i>0; i-- )
                
            {
                int j = rand()%(i+1);
                int tmp = rand_vec(j,0);
                rand_vec(j,0) = rand_vec(i,0);
                rand_vec(i,0) = tmp;
            }
			for(int i=0; i<k; i++)
            
            {
                s_ind[i] = rand_vec(i,0);
			}
                }

                
                        
                
         MPI_Bcast(s_ind, s, MPI_UINT64_T, 0, MPI_COMM_WORLD); // data type for MPI
		 MPI_Barrier(MPI_COMM_WORLD);
		 
		
		for(int i=0; i< k; i++)
		{
			for(int j=0; j< k; j++)
			{
				w(i,j)= kernelf(A.col(s_ind[i]), A.col(s_ind[j]),sig);
				}
			}
			
			
		
			invw = w.inverse();
			

			
			
		/*compute local C*/
		for(int j=0; j< k; j++)
		{
			for(uint64_t i = 0; i < myn ; i++)
			{
				C(i,j)= kernelf((A.col(mystartid+i)),(A.col(s_ind[j])),sig);
				}
			}
	     
		/*compute local R*/
         R = invw *(C.transpose());

	     				
		
				
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		
		/*time elapsed*/
		if(!myrank)
			{
				printf( "\033[1;31mNystrom factorization runtime = %fs\033[0m\n",(t2 - tread1) );
			}
		
		/* *****error computation***** */
		double ts1,ts2;
		ts1 =  MPI_Wtime();
		vector<uint64_t> s_indSorted (s_ind,s_ind+k);
		sort (s_indSorted.begin(), s_indSorted.begin()+k);
		ts2 =  MPI_Wtime();
		if(!myrank)
			{
				printf( "\033[1;31mFull sort time = %fs\033[0m\n",(ts2 - ts1) );
			}
		

		ts1 =  MPI_Wtime();
		if(!myrank)
			printf( "\033[32;1mStart calculating error\033[0m\n");
		int numerr= err_countMax;

		
		int err_ind_i, err_ind_j;
			
		 double sumerr [2];
		 sumerr[0] = 0;
		 sumerr[1] = 0;
		 double finalerr[2];
		 double tempa;
		 double  tempb;
		 for(int i=0; i< numerr; i++)
					{   
						err_ind_i= rand()%myn; 
						err_ind_j = rand()%myn; 
					
						tempa = kernelf(A.col(err_ind_i+mystartid),A.col(err_ind_j+mystartid),sig);
						tempb = C.row(err_ind_i)*invw*(C.row(err_ind_j).transpose());
						sumerr [0] =  sumerr[0]+(tempa- tempb)*(tempa- tempb); // norm 2 of the error
						sumerr[1]  = sumerr[1]+ tempa*tempa;// norm 2 of the acual dara
					}						
		 sumerr[0] = sumerr[0];	
		 sumerr[1] = sumerr[1];
		 MPI_Barrier(MPI_COMM_WORLD);								
		 MPI_Allreduce(sumerr, finalerr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
		 ts2 =  MPI_Wtime();	
		 if(!myrank)
		 {
			 
			 printf("Average l2 error =  %f\n",finalerr[0]/(npes*numerr));
			 printf("Normalized l2 error =  %f\n",finalerr[0]/finalerr[1]);
			 printf( "\033[1;31mError computation time = %fs\033[0m\n",(ts2 - ts1) );
			 cout<<"Sorted selected indices="<<endl;
					  for(int i=0; i<k; i++){
						cout<<s_ind[i]<<" ";
						}
						cout<<endl;
					}
					

	MPI_Finalize();


return 0;
}
