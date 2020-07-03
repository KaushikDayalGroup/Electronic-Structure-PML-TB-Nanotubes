//
//  main.cpp
//  Solving Schrodinger Equation in real-space for OS with local defects
//
//
static char help[] = "Tests solving linear system on 0 by 0 matrix.\n\n";

#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <complex>
#include <stdio.h>
#include <petscksp.h>
#include "petscmat.h"
#include "petscvec.h"
using namespace std;
#define _USE_MATH_DEFINES

//--------------------------------------------------------------
/*Declaration of Subroutines*/
//--------------------------------------------------------------
void pre_process( );
void outputs ( );
Mat Hamiltonian ( );
int gcd( int m, int n );
void helical_vec( );
void gaussian_rec( int gp );
double local_potential ( double x[ ] );
double periodic_potential ( double x[ ] );
complex<double> phi_k( int iangle, int kpoint, int iorbit, int ilattice, double x[ ], int deriv );
double basis (double x[ ], double X[ ], int iorbit, int deriv);
complex<double> hopping ( int isite, int jsite, int ksite, int jatom, int iorbit, int jorbit );
complex<double> rhs ( int isite, int jsite, int ksite, int iorbit );
complex<double>	deformation ( double y );
complex<double> interpolate( double x[ ], int deriv );

//----------------------------------------------------------------------------------------
/*Declaration of input and output stream*/
//----------------------------------------------------------------------------------------
ifstream data_in1("input.txt");					//	input data
ifstream data_in2("Eigns(5,2).txt");	//	input data

ofstream data_out1("Unfolded_52.txt");				//	output data for Matlab


ofstream data_out2("Folded52.txt");				//	output data for Matlab

//---------------------------------------------------------
/*Declaration of Variables*/
//---------------------------------------------------------
int Norbit, iangle, Nangle, l1, l2, v1, v2, k0, nKpoint;		/* number of degrees of freedon in each node */
PetscInt TNDF;													/* total number of degrees of freedon */
double *gaus, *w, X0[ 2 ][ 3 ];				/*Gauss points and weights.*/
int Natom, II, kappa0, angle0;													/*total number of atoms*/
double Jacobian, Length, Length1, Length2, theta, theta1, theta2, torsion, T2, circum, Xv[ 3 ];				/*Length of the elements in x (leng[0]) and y (leng[1]) and z (leng[2]) direction*/
const double A = 5.0, leng = 1.39, epsilon = 0.05;
const int gp = 7, Nneighbor = 15, N = 4, NT = 15;

struct Pdata {
	double xcord, ycord, zcord;
	int sites[ 3 ];
	int neighbors[ Nneighbor ][ 4 ];
};
struct Kdata {
	double k, *E;
	complex<double> **a;
};

Pdata *atoms;				/*Stores nodes coordinates*/
Kdata **bloch;
Vec H0, psi_vec;

const complex<double> I = complex<double> (0,1);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
	Mat	H;
	PC		pc;
	KSP   ksp;
	
	//------------------------------------------------------------
	/*initialize PETSC variables*/
	//-----------------------------------------------------------
	PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(PETSC_USE_COMPLEX)
	SETERRQ(MPI_COMM_SELF,1,"This example requires complex numbers");
#endif
	
	//-------------------------------------------------------------------------------------------------
	/*Get the initial data and genarate the mesh*/
	//-------------------------------------------------------------------------------------------------
	pre_process( );
	gaussian_rec( gp );
	
	H = Hamiltonian ();
	VecDuplicate(H0,&psi_vec);
//	MatView(H,PETSC_VIEWER_STDOUT_WORLD);
//	VecView(H0,PETSC_VIEWER_STDOUT_WORLD);
	cout<<"Stiffness Assembly Done!\n";
	//-----------------------------------------------------------
	/* solve linearized system */
	//----------------------------------------------------------
	
	KSPCreate(MPI_COMM_SELF,&ksp);
     KSPSetType(ksp,KSPPREONLY);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCLU);
	KSPSetOperators(ksp,H,H);
	KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	KSPSetFromOptions(ksp);
	KSPSolve(ksp,H0,psi_vec);
	
	cout<<"Matrix Digonalization Done!\n";
//	VecView( psi_vec,PETSC_VIEWER_STDOUT_WORLD);
	
	//--------------------------
	/*finalizing*/
	//--------------------------
	
	outputs();
	KSPDestroy(&ksp);
	VecDestroy(&H0);
	MatDestroy(&H);
	VecDestroy(&psi_vec);
	PetscFinalize();
	
	return 0;
}

//----------------------------------------------------------------
/*mesh generator subroutine*/
//---------------------------------------------------------------
void pre_process( ) {
	
	int AtomCounter = -1, flag, counter1, counter2, latom, iatom;
	double real_part, imag_part, r, X_c, Y_c, dist2, coord[ 3 ], dummy;
	
	/*read input data*/
    data_in2>>Norbit;
    data_in2>>l1;
    data_in2>>l2;
    data_in2>>Nangle;
    data_in2>>nKpoint;
	data_in1>>Length1;
	data_in1>>Length2;
	data_in1>>k0;
	data_in1>>angle0;
	
	/*assign the size of dynamic arrays*/
	r = leng / ( 2. * M_PI ) * sqrt( 3. * ( l1 * l1 + l2 * l2 + l1 * l2 ) );
	circum = 2. * M_PI * r;
	
	helical_vec( );
        torsion= 0.0*M_PI/180;
	theta = M_PI / 6. - atan( sqrt( 3.0 ) * l2 / ( 2. * l2 + l1 ) );
	theta1 = 2.0 * M_PI / double( gcd( l1, l2 ) );
	theta2 = M_PI * ( v2 * ( 2 * l2 + l1 ) + v1 * ( l2 + 2 * l1 ) ) / double( l1 * l1 + l2 * l2 + l1 * l2 );
	theta2= theta2 + torsion* T2;
	cout<<"theta1 = "<<theta1<<"\ttheta2 = "<<theta2<<"\ttheta = "<<theta<<endl;
	
	X0[ 0 ][ 0 ] = r;			X0[ 0 ][ 1 ] = 0.0;	X0[ 0 ][ 2 ]= 0.0;
	X0[ 1 ][ 0 ] = r * cos( M_PI * ( l1 + l2 ) / ( l1 * l1 + l2 * l2 + l1 * l2 ) );		X0[ 1 ][ 1 ] = r * sin( M_PI * ( l1 + l2 ) / ( l1 * l1 + l2 * l2 + l1 * l2 ) );	X0[ 1 ][ 2 ] = leng * ( l2 - l1 ) / ( 2. * sqrt( l1 * l1 + l2 * l2 + l1 * l2 ) );
	
	T2 = 3.0 * leng * gcd( l1, l2 ) / ( 2.0 * sqrt( l1 * l1 + l2 * l2 + l1 * l2 ) );
	II = int( ( Length1 + 2 * Length2 ) / T2 );
	if ( II / 2 == double ( II ) / 2. ) {
		II ++;
	}
	
	Length = ( II - 1 ) * T2 - X0[ 1 ][ 2 ];
//-!-!-!!!!!!!!!!!!_!_!_!_!__!_!_!_!__!__!_!_!_!__!___!_!_!__!_!_!_!__!_!_!_!__!_!_!_!_!__!_!_!_______________!_!_!_!_!_!__!_!_!___!_!_!_!_!_______!!!!!!!!__!)!_!!_!)!_!__!_!_!-1
	Natom = 2 * II * gcd( l1, l2 );
	cout<<"Circumfrence = "<<circum<<"\tLength = "<<Length<<"\tII = "<<II<<"\td = "<<gcd( l1, l2 )<<"\t Natom = "<<Natom<<endl;
	atoms = new Pdata[ Natom ];
    bloch = new Kdata*[ Nangle ];
    
    /*input Bloch information*/
    for ( int iangle = 0; iangle < Nangle; iangle ++ ) {
        
        bloch[ iangle ] = new Kdata[ 2 * nKpoint - 1 ];
        for ( int kpoint = 0; kpoint < 2 * nKpoint - 1; kpoint ++ ) {
            
            bloch[ iangle ][ kpoint ].a = new complex<double> *[ 2 * Norbit ];
            bloch[ iangle ][ kpoint ].E = new double[ 2 * Norbit ];
            data_in2>>bloch[ iangle ][ kpoint ].k>>dummy;
            
            for (int iorbit = 0; iorbit < 2 * Norbit; iorbit ++) {
                data_in2>>bloch[ iangle ][ kpoint ].E[ iorbit ];
                bloch[ iangle ][ kpoint ].a[ iorbit ] = new complex<double> [ 2 * Norbit ];
                for (int jorbit = 0; jorbit < 2 * Norbit; jorbit ++) {
                    data_in2>>real_part>>imag_part;
                    bloch[ iangle ][ kpoint ].a[ iorbit ][ jorbit ] = complex<double> ( real_part, imag_part );
                }
            }
        }
    }
	
	cout<<"Incident Wave: k = "<<bloch[ angle0 ][ k0 ].k<<"\tE = "<<bloch[ angle0 ][ k0 ].E[ 0 ]<<endl;
	/*input the atoms coordinates*/
		X_c =  X0[ 0 ][ 0 ] * cos( M_PI/2. ) - X0[ 0 ][ 1 ] * sin( M_PI/2. );
		Y_c =  X0[ 0 ][ 0 ] * sin( M_PI/2. ) + X0[ 0 ][ 1 ] * cos( M_PI/2. );
				
	for ( int i = 0; i < II; i ++ ) {
		
		for ( int j = 0; j < gcd( l1, l2 ); j ++ ) {
			
			for ( int k = 0; k < 2; k ++ ) {
			
				//if (X_c > = 0 && Y_c > = 0)
				{
				AtomCounter ++;
							
				atoms[ AtomCounter ].xcord = X0[ k ][ 0 ] * cos( i * theta2 + j * theta1 ) - X0[ k ][ 1 ] * sin( i * theta2 + j * theta1 );
				atoms[ AtomCounter ].ycord = X0[ k ][ 0 ] * sin( i * theta2 + j * theta1 ) + X0[ k ][ 1 ] * cos( i * theta2 + j * theta1 );
				atoms[ AtomCounter ].zcord = X0[ k ][ 2 ] + i * T2;
				
				atoms[ AtomCounter ].sites[0] = i;
				atoms[ AtomCounter ].sites[1] = j;
				atoms[ AtomCounter ].sites[2] = k;
			}
			
			}
		}
	}
	Natom = AtomCounter +1;	
    kappa0 = ( II + 1 ) / 2 - 1;
    iatom = kappa0 * 2 * gcd( l1, l2 ) + 2 * 2 + 0;
    coord[ 0 ] = atoms[ iatom ].xcord;
    coord[ 1 ] = atoms[ iatom ].ycord;
    coord[ 2 ] = atoms[ iatom ].zcord;
    cout<<"Xv = "<<coord[ 0 ]<<"\tYv = "<<coord[ 1 ]<<"\tZv = "<<coord[ 2 ]<<endl;
    //data_out2<<"Xv = "<<coord[ 0 ]<<"\tYv = "<<coord[ 1 ]<<"\tZv = "<<coord[ 2 ]<<endl;
    Xv[ 0 ] = coord[ 0 ];	Xv[ 1 ] = coord[ 1 ]; Xv[ 2 ] = coord[ 2 ];
	/*	find each atom's neighbors */
	int No_of_nei[ Natom ];
	
	for (int iii = 0; iii < Natom; iii ++ ){
		No_of_nei[ iii ] = 0;
	} 
	
	for ( AtomCounter = 0; AtomCounter < Natom; AtomCounter ++ ) {
		

		for (int ii = 0; ii <Natom; ii ++ ) {
			dist2= ( atoms[ AtomCounter ].xcord - atoms[ii ].xcord ) * ( atoms[ AtomCounter ].xcord - atoms[ ii ].xcord ) + (atoms[ AtomCounter ].ycord - atoms[ ii ].ycord ) * ( atoms[ AtomCounter ].ycord - atoms[ ii ].ycord ) + (atoms[ AtomCounter ].zcord - atoms[ ii ].zcord ) * ( atoms[ AtomCounter ].zcord - atoms[ ii ].zcord );
			if ( dist2 <= 6.1* leng * leng ) {
				
				counter1 = No_of_nei[ AtomCounter ];
				counter2 = No_of_nei[ ii];
				if(counter1 >= Nneighbor-1){
				continue;
				}
				else	{
				atoms[ AtomCounter ].neighbors[ counter1 ][ 0 ] = atoms[ ii ].sites[ 0 ];
				atoms[ AtomCounter ].neighbors[ counter1 ][ 1 ] = atoms[ ii ].sites[ 1 ];
				atoms[ AtomCounter ].neighbors[ counter1 ][ 2 ] = atoms[ ii ].sites[ 2 ];
				atoms[ AtomCounter ].neighbors[ counter1 ][ 3 ] = ii;
				
			//	atoms[ ii ].neighbors[ counter2 ][ 0 ] = atoms[ AtomCounter ].sites[ 0 ];
			//	atoms[ ii ].neighbors[ counter2 ][ 1 ] = atoms[ AtomCounter ].sites [ 1 ];
			//	atoms[ ii ].neighbors[ counter2 ][ 2 ] = atoms[ AtomCounter ].sites [ 2 ];
			//	atoms[ ii ].neighbors[ counter2 ][ 3 ] =  AtomCounter;
				
				No_of_nei[ AtomCounter ] = No_of_nei[ AtomCounter ] + 1;
				}

			//	No_of_nei[ ii ] = No_of_nei[ ii ] + 1;
			}
		}
		
	}
	
//	AtomCounter = -1;
//	for ( int i = 0; i < II; i ++ ) {
//		
//		for ( int j = 0; j < gcd( l1, l2 ); j ++ ) {
//			
//			for ( int k = 0; k < 2; k ++ ) {
//				
//				AtomCounter ++;
//				for ( int ineighbor = 0; ineighbor < Nneighbor; ineighbor ++ ) {
//					atoms[ AtomCounter ].neighbors[ ineighbor ][ 0 ] = -10;	atoms[ AtomCounter ].neighbors[ ineighbor ][ 1 ] = -1;
//				}
//				flag = 0;
//				counter = -1;
//				
//				//				cout<<AtomCounter;
//				for ( iatom = -NT; iatom < NT + 1; iatom ++ ) {
//					
//					if ( flag ==1 ) {
//						break;
//					}
//					
//					for ( int jatom = - (gcd( l1, l2 ) - 1 ) / 2; jatom < (gcd( l1, l2 ) - 1 ) / 2 + 1; jatom ++ ) {
//						
//						if ( flag ==1 ) {
//							break;
//						}
//						
//						for ( int katom = 0; katom < 2; katom ++ ) {
//							
//							if ( jatom + j > gcd( l1, l2 ) - 1 ) {
//								latom = ( iatom + i ) * gcd( l1, l2 ) * 2 + ( j + jatom - gcd( l1, l2 ) ) * 2 + katom;
//							}
//							else if ( jatom + j < 0 ) {
//								latom = ( iatom + i ) * gcd( l1, l2 ) * 2 + ( j + jatom + gcd( l1, l2 ) ) * 2 + katom;
//							}
//							else {
//								latom = ( iatom + i ) * gcd( l1, l2 ) * 2 + ( j + jatom ) * 2 + katom;
//							}
//							
//							if ( latom < 0 or latom > Natom ) {
//								break;
//							}
//							
//							if ( ( atoms[ AtomCounter ].xcord - atoms[ latom ].xcord ) * ( atoms[ AtomCounter ].xcord - atoms[ latom ].xcord ) + ( atoms[ AtomCounter ].ycord - atoms[ latom ].ycord ) * ( atoms[ AtomCounter ].ycord - atoms[ latom ].ycord ) + ( atoms[ AtomCounter ].zcord - atoms[ latom ].zcord ) * ( atoms[ AtomCounter ].zcord - atoms[ latom ].zcord ) < 6.1 * leng * leng ) {
//								counter ++;
//								atoms[ AtomCounter ].neighbors[ counter ][ 0 ] = iatom + i;
//								if ( jatom + j > gcd( l1, l2 ) - 1 ) {
//									atoms[ AtomCounter ].neighbors[ counter ][ 1 ] = jatom + j - gcd( l1, l2 );
//								}
//								else if ( jatom + j < 0 ) {
//									atoms[ AtomCounter ].neighbors[ counter ][ 1 ] = jatom + j + gcd( l1, l2 );
//								}
//								else {
//									atoms[ AtomCounter ].neighbors[ counter ][ 1 ] = jatom + j;
//								}
//								atoms[ AtomCounter ].neighbors[ counter ][ 2 ] = katom;
////								cout<<"\t\t"<<latom<<"\t"<<atoms[ AtomCounter ].neighbors[ counter ][ 0 ]<<"\t"<<atoms[ AtomCounter ].neighbors[ counter ][ 1 ]<<"\t"<<atoms[ AtomCounter ].neighbors[ counter ][ 2 ];
//								if ( counter == Nneighbor - 1 ) {
//									flag = 1;
//									break;
//								}
//							}
//						}
//					}
//				}
//				//				cout<<endl;
//			}
//		}
//	}
}

//----------------------------------------------
/*Determine v1 & v2*/
//----------------------------------------------
void helical_vec( ) {
	
	int t1 = ( 2 * l2 + l1 ) / gcd( 2 * l2 + l1, 2 * l1 + l2 ), t2 = - ( 2 * l1 + l2 ) / gcd( 2 * l2 + l1, 2 * l1 + l2 ), N0 = 4 * ( l1 * l1 + l2 * l2 + l1 * l2 ) / gcd( 2 * l2 + l1, 2 * l1 + l2 );
	
	if ( l2 != 0 ) {
		for ( int i = - 1; i < l2 + 1; i ++ ) {
			if ( abs( int( double( gcd( l1, l2 ) + i * l1 ) / double( l2 ) ) - ( gcd( l1, l2 ) + i * l1 ) / l2 ) < 1e-5 ) {
				v2 = i;
				v1 = ( gcd( l1, l2 ) + i * l1 ) / l2;
				if ( t1 * v2 - t2 * v1 > 0 and t1 * v2 - t2 * v1 < N0 / gcd( l1, l2 ) ) {
					break;
				}
			}
		}
	}
	else {
		v2 = -1;
		for ( int i = - 2; i < 3; i ++ ) {
			v1 = i;
			if ( t1 * v2 - t2 * v1 > 0 and t1 * v2 - t2 * v1 < N0 / gcd( l1, l2 ) ) {
				break;
			}
		}
	}
	cout<<"v1 = "<<v1<<", v2 = "<<v2<<endl;
}

//----------------------------------------------
/*Determine l1 & l2*/
//----------------------------------------------
int gcd( int m, int n ) {
	
	int  G;
	for( int i = 1;i <= m and i <= n; i ++ ){
		
		if( m % i == 0 and n % i == 0 ){
			
			G = i;
		}
	}
	
	if ( m == 0 )	G = n;
	if ( n == 0 )	G = m;
	return G;
}

//---------------------------------------------------------
/*Determine Gauss points*/
//---------------------------------------------------------
void gaussian_rec(int gp) {
	switch (gp) {
		case 1:
			gaus=new double[1];
			w=new double[1];
			gaus[0]=0.0;
			w[0]=2.0;
			break;
		case 2:
			gaus=new double[2];
			w=new double[2];
			gaus[0]=-1.0/sqrt(double(3));
			gaus[1]=-gaus[0];
			w[0]=1.0;
			w[1]=1.0;
			break;
			
		case 3:
			gaus=new double[3];
			w=new double[3];
			gaus[0] = -sqrt(double(3))/5.0;
			gaus[2] = -gaus[0];
			gaus[1] = 0;
			w[0] = 5.0/9.0;
			w[1] = 8.0/9.0;
			w[2] = 5.0/9.0;
			break;
			
		case 5:
			gaus=new double[5];
			w=new double[5];
			gaus[0] = - sqrt( 5.0 + 2.0 * sqrt( 10.0 / 7.0 ) ) / 3.0;
			gaus[1] = - sqrt( 5.0 - 2.0 * sqrt( 10.0 / 7.0 ) ) / 3.0;
			gaus[2] = 0;
			gaus[3] = - gaus[1];
			gaus[4] = - gaus[0];
			w[0] = ( 322.0 - 13.0 * sqrt( 70 ) ) / 900.0;
			w[1] = ( 322.0 + 13.0 * sqrt( 70 ) ) / 900.0;
			w[2] = 128.0 / 225.0;
			w[3] = ( 322.0 + 13.0 * sqrt( 70 ) ) / 900.0;
			w[4] = ( 322.0 - 13.0 * sqrt( 70 ) ) / 900.0;
			break;
			
		case 7:
			gaus=new double[ 7 ];
			w=new double[ 7 ];
			gaus[ 0 ] = - 0.9491079123427585;		w[ 0 ] = 0.1294849661688697;
			gaus[ 1 ] = - 0.7415311855993945;		w[ 1 ] = 0.2797053914892766;
			gaus[ 2 ] = - 0.4058451513773972;		w[ 2 ] = 0.3818300505051189;
			gaus[ 3 ] = 0.0000000000000000;		w[ 3 ] = 0.4179591836734694;
			gaus[ 4 ] = 0.4058451513773972;		w[ 4 ] = 0.3818300505051189;
			gaus[ 5 ] = 0.7415311855993945;		w[ 5 ] = 0.2797053914892766;
			gaus[ 6 ] = 0.9491079123427585;		w[ 6 ] = 0.1294849661688697;
			break;
			
		case 9:
			gaus=new double[ 9 ];
			w=new double[ 9 ];
			gaus[ 0 ] = - 0.9681602395076261;		w[ 0 ] = 0.0812743883615744;
			gaus[ 1 ] = - 0.8360311073266358;		w[ 1 ] = 0.1806481606948574;
			gaus[ 2 ] = - 0.6133714327005904;		w[ 2 ] = 0.2606106964029354;
			gaus[ 3 ] = - 0.3242534234038089;		w[ 3 ] = 0.3123470770400029;
			gaus[ 4 ] = 0.0000000000000000;		w[ 4 ] = 0.3302393550012598;
			gaus[ 5 ] = 0.3242534234038089;		w[ 5 ] = 0.3123470770400029;
			gaus[ 6 ] = 0.6133714327005904;		w[ 6 ] = 0.2606106964029354;
			gaus[ 7 ] = 0.8360311073266358;		w[ 7 ] = 0.1806481606948574;
			gaus[ 8 ] = 0.9681602395076261;		w[ 8 ] = 0.0812743883615744;
			break;
			
		case 11:
			gaus=new double[ 11 ];
			w=new double[ 11 ];
			gaus[ 0 ] = - 0.9782286581460570;		w[ 0 ] = 0.0556685671161737;
			gaus[ 1 ] = - 0.8870625997680953;		w[ 1 ] = 0.1255803694649046;
			gaus[ 2 ] = - 0.7301520055740494;		w[ 2 ] = 0.1862902109277343;
			gaus[ 3 ] = - 0.5190961292068118;		w[ 3 ] = 0.2331937645919905;
			gaus[ 4 ] = - 0.2695431559523450;		w[ 4 ] = 0.2628045445102467;
			gaus[ 5 ] = 0.0000000000000000;		w[ 5 ] = 0.2729250867779006;
			gaus[ 6 ] = 0.2695431559523450;		w[ 6 ] = 0.2628045445102467;
			gaus[ 7 ] = 0.5190961292068118;		w[ 7 ] = 0.2331937645919905;
			gaus[ 8 ] = 0.7301520055740494;		w[ 8 ] = 0.1862902109277343;
			gaus[ 9 ] = 0.8870625997680953;		w[ 9 ] = 0.1255803694649046;
			gaus[ 10 ] = 0.9782286581460570;		w[ 10 ] = 0.0556685671161737;
			break;
			
		case 15:
			gaus=new double[ 15 ];
			w=new double[ 15 ];
			gaus[ 0 ] = - 0.9879925180204854;		w[ 0 ] = 0.0307532419961173;
			gaus[ 1 ] = - 0.9372733924007060;		w[ 1 ] = 0.0703660474881081;
			gaus[ 2 ] = - 0.8482065834104272;		w[ 2 ] = 0.1071592204671719;
			gaus[ 3 ] = - 0.7244177313601701;		w[ 3 ] = 0.1395706779261543;
			gaus[ 4 ] = - 0.5709721726085388;		w[ 4 ] = 0.1662692058169939;
			gaus[ 5 ] = - 0.3941513470775634;		w[ 5 ] = 0.1861610000155622;
			gaus[ 6 ] = - 0.2011940939974345;		w[ 6 ] = 0.1984314853271116;
			gaus[ 7 ] = 0.0000000000000000;		w[ 7 ] = 0.2025782419255613;
			gaus[ 8 ] = 0.2011940939974345;		w[ 8 ] = 0.1984314853271116;
			gaus[ 9 ] = 0.3941513470775634;		w[ 9 ] = 0.1861610000155622;
			gaus[ 10 ] = 0.5709721726085388;		w[ 10 ] = 0.1662692058169939;
			gaus[ 11 ] = 0.7244177313601701;		w[ 11 ] = 0.1395706779261543;
			gaus[ 12 ] = 0.8482065834104272;		w[ 12 ] = 0.1071592204671719;
			gaus[ 13 ] = 0.9372733924007060;		w[ 13 ] = 0.0703660474881081;
			gaus[ 14 ] = 0.9879925180204854;		w[ 14 ] = 0.0307532419961173;
			break;
			
		case 21:
			gaus=new double[ 21 ];
			w=new double[ 21 ];
			gaus[ 0 ] = - 0.9937521706203895;		w[ 0 ] = 0.0160172282577743;
			gaus[ 1 ] = - 0.9672268385663063;		w[ 1 ] = 0.0369537897708525;
			gaus[ 2 ] = - 0.9200993341504008;		w[ 2 ] = 0.0571344254268572;
			gaus[ 3 ] = - 0.8533633645833173;		w[ 3 ] = 0.0761001136283793;
			gaus[ 4 ] = - 0.7684399634756779;		w[ 4 ] = 0.0934444234560339;
			gaus[ 5 ] = - 0.6671388041974123;		w[ 5 ] = 0.1087972991671484;
			gaus[ 6 ] = - 0.5516188358872198;		w[ 6 ] = 0.1218314160537285;
			gaus[ 7 ] = - 0.4243421202074388;		w[ 7 ] = 0.1322689386333375;
			gaus[ 8 ] = - 0.2880213168024011;		w[ 8 ] = 0.1398873947910731;
			gaus[ 9 ] = - 0.1455618541608951;		w[ 9 ] = 0.1445244039899700;
			gaus[ 10 ] = 0.0000000000000000;		w[ 10 ] = 0.1460811336496904;
			gaus[ 11 ] = 0.1455618541608951;		w[ 11 ] = 0.1445244039899700;
			gaus[ 12 ] = 0.2880213168024011;		w[ 12 ] = 0.1398873947910731;
			gaus[ 13 ] = 0.4243421202074388;		w[ 13 ] = 0.1322689386333375;
			gaus[ 14 ] = 0.5516188358872198;		w[ 14 ] = 0.1218314160537285;
			gaus[ 15 ] = 0.6671388041974123;		w[ 15 ] = 0.1087972991671484;
			gaus[ 16 ] = 0.7684399634756779;		w[ 16 ] = 0.0934444234560339;
			gaus[ 17 ] = 0.8533633645833173;		w[ 17 ] = 0.0761001136283793;
			gaus[ 18 ] = 0.9200993341504008;		w[ 18 ] = 0.0571344254268572;
			gaus[ 19 ] = 0.9672268385663063;		w[ 19 ] = 0.0369537897708525;
			gaus[ 20 ] = 0.9937521706203895;		w[ 20 ] = 0.0160172282577743;
			break;
	}
}

//-------------------------------------------------------------------------------------------------------------------------
/*calculate the basis and its derivative at x-X0 and y-Y0*/
//------------------------------------------------------------------------------------------------------------------------
double basis (double x[ ], double X[ ], int iorbit, int deriv) {
	
	double func = 0.0, alpha, r2 = 0.0;
	int n;
	
	for ( int i = 0; i < 3; i ++ ) {
		r2 += ( x[ i ] - X[ i ] ) * ( x[ i ] - X[ i ] );
	}
	
	switch ( iorbit ) {
		case 0:
			alpha = 2.;	n = 0;
			break;
		case 1:
			alpha = 1.;	n = 0;
			break;
		case 2:
			alpha = 1.;	n = 1;
			break;
		case 3:
			alpha = 1. / 2;	n = 1;
			break;
		case 4:
			alpha = 1. / 2;	n = 0;
			break;
		case 5:
			alpha = 1. / 2;	n = 3;
			break;
		case 6:
			alpha = 1. / 4;	n = 2;
			break;
	}
	
	alpha /= leng;
	
	if ( deriv == 0 ) {
		func = exp( - alpha * r2 );
	}
	else if ( deriv < 4 ) {
		func = exp( - alpha * r2 ) * ( x[ deriv - 1 ] - X[ deriv - 1 ] ) * ( - 2.0 * alpha );
	}
    else {
        func = 2. * alpha * exp( - alpha * r2 ) * ( 2.0 * alpha * r2 - 3. );
    }
	
	return func;
}

//----------------------------------------------------------------------------
/* interpolates the result at point x*/
//----------------------------------------------------------------------------
complex<double> interpolate( double x[ ], int deriv ) {
	
	complex<double> func = 0.;
	int iatom;
	double r2, coord[ 3 ], psi;
	PetscScalar *temp;
	
	int isite0 = int( x[ 2 ] / T2 );
	
	VecGetArray(psi_vec, &temp);
	
	for ( int isite = isite0 - NT; isite < isite0 + NT + 1; isite ++ ) {
		
		for ( int jsite = 0; jsite < gcd( l1, l2 ); jsite ++ ) {
			
			for ( int ksite = 0; ksite < 2; ksite ++ ) {
				
				iatom = isite * 2 * gcd( l1, l2 ) + 2 * jsite + ksite;
                if ( iatom < 0 or iatom > Natom ) {
                    continue;
                }
                coord[ 0 ] = atoms[ iatom ].xcord;
                coord[ 1 ] = atoms[ iatom ].ycord;
                coord[ 2 ] = atoms[ iatom ].zcord;
				
				r2 = 0.0;
				for ( int i = 0; i < 3; i ++ ) {
					r2 += ( x[ i ] - coord[ i ] ) * ( x[ i ] - coord[ i ] );
				}
				
				if ( r2 < 16.1 * leng * leng ) {
					for ( int iorbit = 0; iorbit < Norbit; iorbit ++ ) {
						
						psi = basis( x, coord, iorbit, deriv );
						func += temp[ Norbit * iatom + iorbit ] * psi;
					}
				}
			}
		}
	}
	
	VecRestoreArray(psi_vec, &temp);
	
	return func;
}

//----------------------------------------------------------------------------
/*calculate the Bloch wavefunction*/
//----------------------------------------------------------------------------
complex<double> phi_k( int iangle, int kpoint, int iorbit, int ilattice, double x[ ], int deriv ) {
	
	complex<double> func = 0.0;
	double coord[ 3 ], psi, r2;
	int isite0 = int( x[ 2 ] / T2 );
	
	for ( int isite = isite0 - NT; isite < isite0 + NT + 1; isite ++ ) {
		
		for ( int jsite = 0; jsite < gcd( l1, l2 ); jsite ++ ) {
			
			for ( int ksite = 0; ksite < 2; ksite ++ ) {
				
                coord[ 0 ] = X0[ ksite ][ 0 ] * cos( isite * theta2 + jsite * theta1 ) - X0[ ksite ][ 1 ] * sin( isite * theta2 + jsite * theta1 );
                coord[ 1 ] = X0[ ksite ][ 0 ] * sin( isite * theta2 + jsite * theta1 ) + X0[ ksite ][ 1 ] * cos( isite * theta2 + jsite * theta1 );
                coord[ 2 ] = X0[ ksite ][ 2 ] + isite * T2;
				
				r2 = 0.0;
				for ( int i = 0; i < 3; i ++ ) {
					r2 += ( x[ i ] - coord[ i ] ) * ( x[ i ] - coord[ i ] );
				}
				
				if ( r2 < 16.1 * leng * leng ) {
					
					for ( int jorbit = 0; jorbit < Norbit; jorbit ++ ) {
						
						psi = basis( x, coord, iorbit, deriv );
						func += bloch[ iangle ][ kpoint ].a[ 2 * iorbit + ilattice ][ 2 * jorbit + ksite ] * exp( I * ( bloch[ iangle ][ kpoint ].k * T2 * isite + iangle * ( theta1 * jsite + theta2 * isite ) ) ) * psi;
					}
				}
			}
		}
	}
	
	return func;
}

//-------------------------------------------------------------------------------------------
/*calculate the potential at x-X0 and y-Y0*/
//-------------------------------------------------------------------------------------------
double local_potential ( double x[ ] ) {
	
    double func = 0.0, sigma = 3.0 / leng, r2 = 0.0;
	
	for ( int i = 0; i < 3; i ++ ) {
		r2 += ( x[ i ] - Xv[ i ] ) * ( x[ i ] - Xv[ i ] );
	}
	
	if ( r2 < 16.1 * leng * leng ) {
		func = - A * exp ( - sigma * r2 );
	}
	
	return func;
}

//--------------------------------------------------------------------------------------------------------------------------------
/*calculate the potential and its derivative at x-X0 and y-Y0*/
//--------------------------------------------------------------------------------------------------------------------------------
double periodic_potential ( double x[ ] ) {
	
	double func = 0.0, sigma = 3.0 / leng, r2, coord[ 3 ];
	int isite0;
    
	isite0 = int( x[ 2 ] / T2 );
	
	for ( int isite = isite0 - NT; isite < isite0 + NT + 1; isite ++ ) {
		
		for ( int jsite = 0; jsite < gcd( l1, l2 ); jsite ++ ) {
			
			for ( int ksite = 0; ksite < 2; ksite ++ ) {
				
                coord[ 0 ] = X0[ ksite ][ 0 ] * cos( isite * theta2 + jsite * theta1 ) - X0[ ksite ][ 1 ] * sin( isite * theta2 + jsite * theta1 );
                coord[ 1 ] = X0[ ksite ][ 0 ] * sin( isite * theta2 + jsite * theta1 ) + X0[ ksite ][ 1 ] * cos( isite * theta2 + jsite * theta1 );
                coord[ 2 ] = X0[ ksite ][ 2 ] + isite * T2;
				
				r2 = 0.0;
				for ( int i = 0; i < 3; i ++ ) {
					r2 += ( x[ i ] - coord[ i ] ) * ( x[ i ] - coord[ i ] );
				}
				
				if ( r2 < 16.1 * leng * leng ) {
					
					func += A * exp ( - sigma * r2 );
				}
			}
		}
	}
	
	return func;
}

//-----------------------------------------------------------------------------------
/* calculate the deformation gradiant*/
//-----------------------------------------------------------------------------------
complex<double>	deformation ( double y ) {
	
	complex<double>F;
	double sigma0 = epsilon * Length2, sigma = 0.0, dist;
	
	if ( y > Length - Length2 + X0[ 1 ][ 2 ] ) {
		dist = y - Length + Length2 + X0[ 1 ][ 2 ];
		sigma = 30 * sigma0 * dist * dist * ( Length2 - dist ) * ( Length2 - dist ) / ( Length2 * Length2 * Length2 * Length2 * Length2 );
	}
	else if ( y < Length2 + X0[ 1 ][ 2 ] ) {
		dist = Length2 + X0[ 1 ][ 2 ] - y;
		sigma = 30 * sigma0 * dist * dist * ( Length2 - dist ) * ( Length2 - dist ) / ( Length2 * Length2 * Length2 * Length2 * Length2 );
	}
	
	F = 1.0 + I * sigma;
	
	return 1.0 / F;
}

//----------------------------------------------------------------------------------------------------------
/*calculate hopping integral between two nodes*/
//----------------------------------------------------------------------------------------------------------
complex<double> hopping ( int isite, int jsite, int ksite, int ineighbor, int iorbit, int jorbit ) {
	
	complex<double> func = 0.0, InvF = 1.0, phi, dy_phi;
	double x[ 3 ], Vp = 0., Vd = 0., psi[ 2 ], dpsi[ 2 ][ 3 ], Ek = bloch[ angle0 ][ k0 ].E[ 0 ], Icoord[ 3 ], Jcoord[ 3 ], Y0[ 3 ][ 2 ];
	int iatom = isite * 2 * gcd( l1, l2 ) + 2 * jsite + ksite, jatom = atoms[ iatom ].neighbors[ ineighbor ][ 0 ] * 2 * gcd( l1, l2 ) + 2 * atoms[ iatom ].neighbors[ ineighbor ][ 1 ] + atoms[ iatom ].neighbors[ ineighbor ][ 2 ];
//    cout<<"\t"<<jatom<<endl;
    Icoord[ 0 ] = atoms[ iatom ].xcord;
    Icoord[ 1 ] = atoms[ iatom ].ycord;
    Icoord[ 2 ] = atoms[ iatom ].zcord;
    Jcoord[ 0 ] = atoms[ jatom ].xcord;
    Jcoord[ 1 ] = atoms[ jatom ].ycord;
    Jcoord[ 2 ] = atoms[ jatom ].zcord;
	
	Y0[ 0 ][ 0 ] = ( Icoord[ 0 ] + Jcoord[ 0 ] ) / 2. - 3. * leng;
	Y0[ 0 ][ 1 ] = ( Icoord[ 0 ] + Jcoord[ 0 ] ) / 2. + 3. * leng;
    Y0[ 1 ][ 0 ] = ( Icoord[ 1 ] + Jcoord[ 1 ] ) / 2. - 3. * leng;
    Y0[ 1 ][ 1 ] = ( Icoord[ 1 ] + Jcoord[ 1 ] ) / 2. + 3. * leng;
	Y0[ 2 ][ 0 ] = ( Icoord[ 2 ] + Jcoord[ 2 ] ) / 2. - 3. * leng;
	Y0[ 2 ][ 1 ] = ( Icoord[ 2 ] + Jcoord[ 2 ] ) / 2. + 3. * leng;
    
	
	if ( Y0[ 2 ][ 0 ] <  0 ) {
		Y0[ 2 ][ 0 ] = 0;
	}
	else if ( Y0[ 2 ][ 1 ] > Length ) {
		Y0[ 2 ][ 1 ] = Length;
	}
	
	Jacobian = 36. * leng * leng * ( Y0[ 2 ][ 1 ] - Y0[ 2 ][ 0 ] ) / ( 8. * N * N * N );
	
	for ( int ielem = 0; ielem < N; ielem ++ ) {
		
		for ( int jelem = 0; jelem < N; jelem ++ ) {
			
			for ( int kelem = 0; kelem < N; kelem ++ ) {
				
				for ( int iGauss = 0; iGauss < gp; iGauss ++ ) {
					
					for ( int jGauss = 0; jGauss < gp; jGauss ++ ) {
						
						for ( int kGauss = 0; kGauss < gp; kGauss ++ ) {
							
							x[ 0 ] = Y0[ 0 ][ 0 ] + ( double( ielem ) + ( gaus[ iGauss ] + 1. ) / 2. ) * 6. * leng / double( N );
							x[ 1 ] = Y0[ 1 ][ 0 ] + ( double( kelem ) + ( gaus[ kGauss ] + 1. ) / 2. ) * 4. * leng / double( N );
							x[ 2 ] = Y0[ 2 ][ 0 ] + ( double( jelem ) + ( gaus[ jGauss ] + 1. ) / 2. ) * ( Y0[ 2 ][ 1 ] - Y0[ 2 ][ 0 ] ) / double( N );
						
							InvF = deformation ( x[ 2 ] );
							psi[ 0 ] = basis( x, Icoord, iorbit, 0 );
							psi[ 1 ] = basis( x, Jcoord, jorbit, 0 );
							
							for ( int i = 1; i < 4; i ++ ) {
								dpsi[ 0 ][ i - 1 ] = basis( x, Icoord, iorbit, i );
								dpsi[ 1 ][ i - 1 ] = basis( x, Jcoord, jorbit, i );
							}
							
							Vp = periodic_potential ( x );
							Vd = local_potential ( x );
							
							for ( int i = 0; i < 2; i++ ) {
                                func += w[ iGauss ] * w[ jGauss ] * w[ kGauss ] * Jacobian * .5 * ( dpsi[ 0 ][ i ] * dpsi[ 1 ][ i ] );
							}
func += w[ iGauss ] * w[ jGauss ] * w[ kGauss ] * Jacobian * .5 * ( dpsi[ 0 ][ 2 ] * dpsi[ 1 ][ 2 ] * InvF );
							
							
							func += w[ iGauss ] * w[ jGauss ] * w[ kGauss ] * Jacobian * psi[ 0 ] * psi[ 1 ] * ( Vp + Vd - Ek ) / InvF;
						}
					}
				}
			}
		}
	}
	
	return func;
}

//---------------------------------------------------------------------------------------------------------
/*calculate hopping integral between two nodes*/
//---------------------------------------------------------------------------------------------------------
complex<double> rhs ( int isite, int jsite, int ksite, int iorbit ) {
	
	complex<double> func = 0.0, phi = 1.0;
	double x[ 3 ], Vd = 1., psi, Icoord[ 3 ], Jcoord[ 3 ], Y0[ 3 ];
	int iatom = isite * 2 * gcd( l1, l2 ) + 2 * jsite + ksite;
	
    Icoord[ 0 ] = atoms[ iatom ].xcord;
    Icoord[ 1 ] = atoms[ iatom ].ycord;
    Icoord[ 2 ] = atoms[ iatom ].zcord;
    Jcoord[ 0 ] = Xv[ 0 ];
    Jcoord[ 1 ] = Xv[ 1 ];
    Jcoord[ 2 ] = Xv[ 2 ];
	
	Y0[ 0 ] = Jcoord[ 0 ] - 3. * leng;
    Y0[ 1 ] = Jcoord[ 1 ] - 3. * leng;
	Y0[ 2 ] = Jcoord[ 2 ] - 3. * leng;
	
	Jacobian = 216 * leng * leng * leng / ( 8. * N * N * N );
	
	if ( ( Icoord[ 0 ] - Jcoord[ 0 ] ) * ( Icoord[ 0 ] - Jcoord[ 0 ] ) + ( Icoord[ 1 ] - Jcoord[ 1 ] ) * ( Icoord[ 1 ] - Jcoord[ 1 ] ) + ( Icoord[ 2 ] - Jcoord[ 2 ] ) * ( Icoord[ 2 ] - Jcoord[ 2 ] ) < 16.1 * leng * leng ) {
		
		for ( int ielem = 0; ielem < N; ielem ++ ) {
			
			for ( int jelem = 0; jelem < N; jelem ++ ) {
				
				for ( int kelem = 0; kelem < N; kelem ++ ) {
					
					for ( int iGauss = 0; iGauss < gp; iGauss ++ ) {
						
						for ( int jGauss = 0; jGauss < gp; jGauss ++ ) {
							
							for ( int kGauss = 0; kGauss < gp; kGauss ++ ) {
								
								x[ 0 ] = Y0[ 0 ] + ( double( ielem ) + ( gaus[ iGauss ] + 1. ) / 2. ) * 6. * leng / double( N );
								x[ 1 ] = Y0[ 1 ] + ( double( kelem ) + ( gaus[ kGauss ] + 1. ) / 2. ) * 6. * leng / double( N );
								x[ 2 ] = Y0[ 2 ] + ( double( jelem ) + ( gaus[ jGauss ] + 1. ) / 2. ) * 6. * leng / double( N );
								
								psi = basis( x, Icoord, iorbit, 0 );
                                phi = phi_k( angle0, k0, 0, 0, x, 0 );
								
								Vd = local_potential ( x );
								
								func += - w[ iGauss ] * w[ jGauss ] * w[ kGauss ] * Jacobian * Vd * psi * phi;
							}
						}
					}
				}
			}
		}
	}

	return func;
}

//-----------------------------------------------------------------------
/* create the Hamiltonian matrix */
//-----------------------------------------------------------------------
Mat Hamiltonian ( ) {
	
	complex<double> s, ss, h[ Nneighbor ][ 2 ];
	int iatom, jatom;
	Mat K;
	PetscScalar Kh, Ks;
	PetscInt idxn, idxm;
	MatCreateSeqAIJ( MPI_COMM_SELF, Natom * Norbit, Natom * Norbit, Nneighbor * Norbit, PETSC_NULL, &K );
	MatSetFromOptions(K);
	MatZeroEntries(K);
	
	VecCreate(MPI_COMM_SELF, &H0);
	VecSetSizes( H0, PETSC_DECIDE, Natom * Norbit );
	VecSetFromOptions(H0);
	VecSet(H0,0.0);
    
	for ( int k = 0; k < 2; k ++ ) {
		
		for (int iorbit = 0; iorbit < Norbit; iorbit ++) {
			
			for (int ineighbor = 0; ineighbor < Nneighbor; ineighbor ++) {
				
				for (int jorbit = 0; jorbit < Norbit; jorbit ++) {
					
					h[ ineighbor ][ k ] = hopping ( II / 3, 0, k, ineighbor, iorbit, jorbit );
					cout<<ineighbor<<"\t"<<h[ ineighbor ][ k ]<<endl;
				}
			}
		}
        cout<<endl;
	}
	
	cout<<"pre-calculations done!\n";
	
	for (int isite = 0; isite < II; isite ++) {
		
		for ( int jsite = 0; jsite < gcd( l1, l2 ); jsite ++ ) {
			
			for ( int ksite = 0; ksite < 2; ksite ++ ) {
				
				iatom = isite * 2 * gcd( l1, l2 ) + 2 * jsite + ksite;
				cout<<iatom<<":\t"<<atoms[ iatom ].zcord<<"\n";
				for (int iorbit = 0; iorbit < Norbit; iorbit ++) {
					
					idxm = Norbit * iatom + iorbit;
					s = rhs( isite, jsite, ksite, iorbit );
					Ks = real( s ) + imag( s ) * PETSC_i;

					VecSetValues( H0, 1, &idxm, &Ks, ADD_VALUES);
                    
					for (int ineighbor = 0; ineighbor < Nneighbor; ineighbor ++) {
						
						jatom = atoms[ iatom ].neighbors[ ineighbor ][ 0 ] * 2 * gcd( l1, l2 ) + 2 * atoms[ iatom ].neighbors[ ineighbor ][ 1 ] + atoms[ iatom ].neighbors[ ineighbor ][ 2 ];
						
						if ( jatom > Natom or jatom < 0 ) {
							continue;
						}
						
						for (int jorbit = 0; jorbit < Norbit; jorbit ++) {
							
							idxn = Norbit * jatom + jorbit;
							
							if ( atoms[ iatom ].zcord < Length2 or atoms[ iatom ].zcord > Length - Length2 or atoms[ jatom ].zcord < Length2 or atoms[ jatom ].zcord > Length - Length2 or abs( s ) > 1.e-10 ) {
								
								ss = hopping ( isite, jsite, ksite, ineighbor, iorbit, jorbit );
							}
							else {
								ss = h[ ineighbor ][ ksite ];
							}
							
							Kh = real( ss ) + imag( ss ) * PETSC_i;
							MatSetValues( K, 1, &idxm, 1, &idxn, &Kh, ADD_VALUES );
						}
					}
//					cout<<endl;
				}
			}
		}
	}

	VecAssemblyBegin(H0);
	VecAssemblyEnd(H0);
	
	MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);
	
	return K;
	MatDestroy(&K);
}

//-------------------------------------------
/*output subroutine*/
//-------------------------------------------
void outputs ( ) {
	
	double Y0[ 3 ], Z0[ 3 ];
	complex<double> func, phi0, conj_func;
	int NN = 30, M = 300;
	PetscScalar *temp;
	
	VecGetArray(psi_vec, &temp);
	
	/*	output for Matlab.txt */
	data_out1<<NN<<"\t"<<M<<"\t"<<leng<<endl<<endl;
	
	for ( int i = 0; i < NN; i ++ ) {
		
		for ( int j = 0; j < M; j ++ ) {
			
			Y0[ 0 ] = - circum / ( 2. * M_PI ) * cos( 2. * M_PI * double( i ) / ( NN - 1 ) );
			Y0[ 1 ] = circum / ( 2. * M_PI ) * sin( 2. * M_PI * double( i ) / ( NN - 1 ) );
			Y0[ 2 ] = double( j ) / double( M - 1 ) * Length;
			Z0[ 0 ] = Y0[ 1 ];
			Z0[ 1 ] = -Y0[ 0 ];
			Z0[ 2 ] = Z0[ 2 ];
			func = interpolate( Y0, 0 );
			
			conj_func = complex<double>( real( func ), - imag( func ) );
			data_out1 << Y0[ 0 ] <<"\t"<<Y0[ 1 ]<<"\t"<< Y0[ 2 ] <<"\t"<< real( func * conj_func )<<"\t";
			phi0 = phi_k( angle0, k0, 0, 0, Y0, 0 );
            func += phi0;
            conj_func = complex<double>( real( func ), -imag( func ) );
            data_out1 <<real( func * conj_func )<<endl;
		}
	}
	
	VecRestoreArray(psi_vec, &temp);
	
}
