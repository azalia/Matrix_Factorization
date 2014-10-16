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
 * Implimentation of OASIS algorithm with a Gaussian Kernel 
 * OASIS is an adaptive column subset selection method that enables factorizing massive kernel matrices: G = C * invw * C^T
 * [m n] = size of matrix A
 * k = number of columns to be selected; [n k] = size of matrix C
 * s = number of initially randomly selected columns
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
        
	   if (s>k)
        {
                if(!myrank)
                        cout << " s (number of initially randomly selected columns) should be less than k (total number of selected columns)"<<endl;
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
		MatrixXd d(myn,1);
		MatrixXd delta(myn,1);
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



		
		/*compute vector of diagonal entries*/
		
		for(uint64_t i=0; i< myn; i++)
                {
                        d(i,0) = kernelf((A.col(i+mystartid)),(A.col(i+mystartid)),sig);
                }

        /*initialize delta*/
		for(uint64_t i=0; i< myn; i++)
                {
                        delta(i,0) = 0;
                }
                
                
		
		
		/*create w by randomly selecting #s input indices*/
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
        for(int i=0; i<s; i++)
        {
            s_ind[i] = rand_vec(i,0);
        }
        }
                    
                        
                
         MPI_Bcast(s_ind, s, MPI_UINT64_T, 0, MPI_COMM_WORLD); // data type for MPI
		 MPI_Barrier(MPI_COMM_WORLD);
		 
		
		for(int i=0; i< s; i++)
		{
			for(int j=0; j< s; j++)
			{
				w(i,j)= kernelf(A.col(s_ind[i]), A.col(s_ind[j]),sig);
				}
			}

			
			//invw.block(0,0,s,s) = w.block(0,0,s,s);
			//invw.block(0,0,s,s) = invw.block(0,0,s,s).inverse();
			
			
			MatrixXd Dhatp = w.block(0,0,s,s);
			JacobiSVD<MatrixXd> svd( w.block(0,0,s,s), ComputeThinU | ComputeThinV);

			double  pinvtoler=1.e-8; // choose your tolerance wisely!
			VectorXd singularValuesinv = svd.singularValues();
			for ( int i=0; i<svd.singularValues().size(); ++i) {
				if ( singularValuesinv(i) > pinvtoler )
				 singularValuesinv(i)=1.0/singularValuesinv(i);
				 else singularValuesinv(i)=0;
			}
			invw.block(0,0,s,s) = (svd.matrixU()*singularValuesinv.asDiagonal()*svd.matrixV().transpose());
			
		/*compute local C_s*/
		for(int j=0; j< s; j++)
		{
			for(uint64_t i = 0; i < myn ; i++)
			{
				C(i,j)= kernelf((A.col(mystartid+i)),(A.col(s_ind[j])),sig);
				}
			}
	     
		/*compute local R_s*/
         R.block(0,0,s,myn) = invw.block(0,0,s,s)*(C.block(0,0,myn,s).transpose());

	     double maxval;
	     double maxval_all;
	     uint64_t  maxid_all;
	     uint64_t maxid , in_s;
	     MatrixXd qk(k,1);

	     				
		 struct maxidval
		 {
		   double val;
		   int rank;
		 };
		 
		 maxidval buff, out;
	   
	     MatrixXd v;
	     int kk;

			t1 = MPI_Wtime(); 
		/* *****start selecting columns***** */
		if(!myrank)
        {cout<<"Start selecting columns"<<endl;
		}
	     while (s<k && maxval_all<.000001)
	     {
	     
						 /*compute delta*/
					 for(uint64_t i=0; i< myn; i++)
							{
									delta(i,0) =  abs(d(i,0)-(C.block(0,0,myn,s).row(i)*R.block(0,0,s,myn).col(i))); // abs value --
							}
					  /*find max delta*/
					  maxval = 0;
					  
					  maxid = 0;
					  in_s = 0;
					  vector<uint64_t> s_indSorted (s_ind,s_ind+s);  
					 
					  sort (s_indSorted.begin(), s_indSorted.begin()+s);  

					  uint64_t i =0; 
					  kk = 0;
					  
					 while(i< myn)
							{
								if(i+mystartid==s_indSorted[kk])
									{
										kk++;
									}
								else if(maxval<delta(i,0))
									{
										maxval = delta(i,0);
										maxid = i+mystartid;
									}
									i++;
							}
							


						buff.rank = myrank;     
						buff.val = maxval;
						
						MPI_Barrier(MPI_COMM_WORLD);
						
						MPI_Allreduce(&buff, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
						MPI_Barrier(MPI_COMM_WORLD);
					  
						maxval_all = out.val;
						if(myrank==out.rank)
							maxid_all = maxid;

						MPI_Barrier(MPI_COMM_WORLD);
						MPI_Bcast(&maxid_all, 1, MPI_UINT64_T, out.rank, MPI_COMM_WORLD); // data type for MPI
						MPI_Barrier(MPI_COMM_WORLD);
						s_ind[s]= maxid_all;
					  

					
					
						/*update C */
						for( uint64_t i = 0; i < myn ; i++){
						    C(i,s) = kernelf(A.col(i+mystartid),(A.col(maxid_all)),sig);
							}
							
					

							
						double sk = 1./maxval_all;	 

						
						for(int i = 0; i < s ; i++){
							qk(i) = kernelf(A.col(s_ind[i]),(A.col(maxid_all)),sig); 
							}
							qk.block(0,0,s,1) = invw.block(0,0,s,s)*qk.block(0,0,s,1);


						invw.block(0,0,s,s) = invw.block(0,0,s,s)+ sk* (qk.block(0,0,s,1)*(qk.block(0,0,s,1).transpose()));
						invw.block(0,s,s,1) = -sk*qk.block(0,0,s,1);
						invw.block(s,0,1,s) = -sk*(qk.block(0,0,s,1).transpose());
						invw(s,s) = sk;			

						/*update R */
						
					   
						R.block(s,0,1,myn) = -sk*((qk.block(0,0,s,1).transpose())*(C.block(0,0,myn,s).transpose()) - C.block(0,s,myn,1).transpose()); 
						
						R.block(0,0,s,myn) =   R.block(0,0,s,myn) - (qk.block(0,0,s,1)*R.block(s,0,1,myn));
				  		
						
					    /* new iteration*/
						s++;
						
						/* progress report (%)*/
						if(!myrank && s%10==0)
							{	
								int prcnt = (s*100)/k;
							printf( "\033[34;1m %%%d complete\033[0m\n",prcnt);
							}		
				}
				
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		
		/*time elapsed*/
		if(!myrank)
			{
				if(s-init_s>0)
					printf( "\033[1;31mElapsed time per col addition = %fs\033[0m\n",(t2 - t1)/(s-init_s) );
				printf( "\033[1;31mTotal time elapsed = %fs\033[0m\n",(t2 - tread1) );
			}
		
		/* *****error computation***** */
		double ts1,ts2;
		ts1 =  MPI_Wtime();
		vector<uint64_t> s_indSorted (s_ind,s_ind+s);
		sort (s_indSorted.begin(), s_indSorted.begin()+s);
		
					
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
						sumerr [0] =  sumerr[0]+(tempa- tempb)*(tempa- tempb); // norm 2 error
						sumerr[1]  = sumerr[1]+ tempa*tempa; // norm 2 of actual kernel matrix
					}						
	
		 MPI_Barrier(MPI_COMM_WORLD);								
		 MPI_Allreduce(sumerr, finalerr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
		 ts2 =  MPI_Wtime();	
		 if(!myrank)
		 {
			 printf("l2 norm of error = %f\n l2 norm of data =  %f\n",finalerr[0],finalerr[1]);
			 printf("Average l2 error =  %f\n",finalerr[0]/(npes*numerr));
			 printf("Average normalized error =  %f\n",finalerr[0]/finalerr[1]);
			 printf( "\033[1;31mError computation time = %fs\033[0m\n",(ts2 - ts1) );
			/* cout<<"Sorted selected indices="<<endl;
					  for(int i=0; i<k; i++){
						cout<<s_ind[i]<<" ";
						}
						cout<<endl;*/
			}

	MPI_Finalize();


return 0;
}
