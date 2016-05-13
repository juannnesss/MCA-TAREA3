#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "omp.h"

#define PI 3.14159265358979323846
#define G 4.492*pow(10,-3)

void random_polar(int N, double r, double P[]);
void polar2cartesian(int N, double P[], double X[]);
void print8columns(int N, double X[], double V[], double U[], double K[], char name[]);
void getAccelerations(int N, double X[], double A[],double eps);
void leapfrog(double dt, int N, double X[], double V[],double A[], double eps);
void pot_energy(int N, double X[], double P[], double U[],double eps);
void kin_energy(int N, double V[], double K[]);

int main (int argc, char **argv)
{
  /*-------------------Variable Initialization----------------------*/
  
  //Parameters
  int N=atoi(argv[1]); //Char argument as number of masses (int)
  double eps=atof(argv[2]); // Prevent infinite accelerations(epsilon)
  
  //Arrays
  double X[3*N]; // Cartesian Coordinates
  double P[3*N]; // Spherical coordinates
  double V[3*N]; // Cartesian Velocities
  //memset(V,0,3*N);
  double A[3*N]; // Cartesian Acceleration
  double U[N];//Potential Energy
  double K[N];//Kinetic Energy
  double Ei[N];//Initial energy
  double Ef[N];//Final Energy
  
  // Dimensions of the space
  double r=0.6205*(pow((double)N,1.0/3)); // Radius of sphere (Polar)

  // Dynamic time, time interval and steps
  double DT=5*pow(G*(N/(4*PI*pow(r,3)/3)),-0.5);
  double dt=pow(G*(N/(4*PI*pow(eps,3)/3)),-0.5);
  double steps = (int)(DT/dt);

  /* ------------------------ Initial calculations-------------------------*/

  /* Create polar coordinate data*/
  random_polar(N,r,P);
  /*Transform to cartesian coordinates*/
  polar2cartesian(N,P,X);
  /*Get Potential Energies*/
  pot_energy(N,X,P,U,eps);
  /*Get kinetic energies*/
  kin_energy(N,V,K);
  /*Print initial conditions*/
  print8columns(N,X,V,U,K,"initial.txt");
  
  /*-----------------------------Evolve calculations ----------------------*/
  /* Calculate accelerations*/
  getAccelerations(N,X,A,eps);
  /* Solve dif eq with leapfrog */
  int l;
  #pragma omp parallel for
  for (l=0;l<steps;l++)
    {
      leapfrog(dt,N,X,V,A,eps);
    }
  /*Get Potential Energies*/
  pot_energy(N,X,P,U,eps);
  /*Get kinetic energies*/
  kin_energy(N,V,K);
  /*Print final conditions*/
  print8columns(N,X,V,U,K,"final.txt");
  

}

void print8columns(int N, double X[], double V[], double U[], double K[], char name[])
{
  FILE * fp;
  int i;
  fp=fopen(name,"w");
  for (i=0;i<N;i++)
    {
      fprintf(fp,"%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\n",X[i],X[i+N],X[i+2*N],V[i],V[i+N],V[i+2*N],K[i],U[i]);
      
    }
  fclose(fp);
}

void polar2cartesian(int N, double P[], double X[])
/*
 Transforms a 3N lenght array of shperical coordinates
of the form [R Th Phi] to a cartesian coordinate array
of the form [X Y Z].
*/ 
{
  int i;
  double R;
  double Th;
  double Phi;
  for (i=0;i<N;i++)
    {
      R=P[i];
      Th=P[i+N];
      Phi=P[i+2*N];
      
      X[i]=R*cos(Th)*sin(Phi);
      X[i+N]=R*sin(Th)*sin(Phi);
      X[i+2*N]=R*cos(Phi);
    }
}

void random_polar(int N, double r, double P[])
/*
 Create random points in spherical coordinates:
The array P is a linear array of length 3N
in which the first N entries are R coordinates,
the next N are polar angle coordinates (Theta),
and the last N entries are azimuthal angles (Phi)
*/
{
  int i;
  srand48(time(NULL)); //random seed
  double max=r;
  int cont=0;
  for (i=0;i<3*N;i++)
    {
      if ((i+1)%N==0)
	{
	  P[i]=max*drand48();
	  if(cont==0)
	    {
	      max=2*PI;
	      cont++;
	    }
	  else if (cont==1)
	    {
	      max=PI;
	    }
	}
      else
	{
	  P[i]=max*drand48();
	}
    }
}

void getAccelerations(int N, double X[], double A[],double eps)
/* 
Calculate acceleration on each particle given the positions
of all the particles calculating the gravitational
force using spherical aproximation
*/
{
  double x,y,z;
  double norm;
  int cont;
  int i,j;
  double ri,rj;
  for (i=0; i<N; i++)
  {
    // Center of mass coordinates (of internal sphere)
    x=0;
    y=0;
    z=0;
    // Number of masses inside the shpere
    cont=0;
    // Radius of particle i
    ri=pow(pow(X[i],2)+pow(X[N+i],2)+pow(X[2*N+i],2),0.5);
    
    // Second loop: Calculate the force of each mass on the particle
    for (j=0;j<N;j++)
      {
	//Radius of particle j
	rj=pow(pow(X[j],2)+pow(X[N+j],2)+pow(X[2*N+j],2),0.5);
	//Do not calculate force on itself & particle inside sphere
	if(j!=i && rj<ri) 
	  {
	    cont++;
	    x+=X[j];
	    y+=X[N+j];
	    z+=X[2*N+j];
	  }
      }
    norm=pow((X[i]-x),2)+pow((X[N+i]-y),2)+pow((X[2*N]-z),2);
    A[i]=cont*G*(X[i]-x)/(pow(norm,1.5)+eps); //X acc
    A[N+i]=cont*G*(X[N+i]-y)/(pow(norm,1.5)+eps);// Y acc
    A[2*N+i]=cont*G*(X[2*N+i]-x)/(pow(norm,1.5)+eps);// Z acc
  }  
}

void leapfrog(double dt, int N, double X[], double V[],double A[], double eps)
/*
Leapfrog solver: gets as parameters arrays for
acceleration, velocities and positions, a distance 
eps and a time interval dt. 
Leapfrog updates velocity and position (One step)
 */
{
  int j;
  for (j=0;j<N;j++)
    {
      //Kick
      V[j]+=0.5*A[j]*dt;//x
      V[j+N]+=0.5*A[j+N]*dt;//y
      V[j+2*N]+=0.5*A[j+2*N]*dt;//z
      //Drift
      X[j]+=V[j]*dt;//x
      X[j+N]+=0.5*A[j+N]*dt;//y
      X[j+2*N]+=0.5*A[j+2*N]*dt;//z
    }
  getAccelerations(N,X,A,eps);
  for (j=0;j<N;j++)
    {
      //Kick
      V[j]+=0.5*A[j]*dt;//x
      V[j+N]+=0.5*A[j+N]*dt;//y
      V[j+2*N]+=0.5*A[j+2*N]*dt;//z
    }
}

void pot_energy(int N, double X[], double P[],double U[], double eps)
/* 
Returns potential energy of each particle given the positions
 */
{
 double x,y,z;
  double norm;
  int cont;
  int i,j;
  double ri,rj;
  for (i=0; i<N; i++)
  {
    // Center of mass coordinates (of internal sphere)
    x=0;
    y=0;
    z=0;
    // Number of masses inside the shpere
    cont=0;
    // Radius of particle i
    ri=pow(pow(X[i],2)+pow(X[N+i],2)+pow(X[2*N+i],2),0.5);
    
    // Second loop: Calculate the force of each mass on the particle
    for (j=0;j<N;j++)
      {
	//Radius of particle j
	rj=pow(pow(X[j],2)+pow(X[N+j],2)+pow(X[2*N+j],2),0.5);
	//Do not calculate force on itself & particle inside sphere
	if(j!=i && rj<ri) 
	  {
	    cont++;
	    x+=X[j];
	    y+=X[N+j];
	    z+=X[2*N+j];
	  }
      }
    norm=pow((X[i]-x),2)+pow((X[N+i]-y),2)+pow((X[2*N]-z),2);
    U[i]=-G*cont/(norm+eps);
  }  
}

void kin_energy(int N, double V[], double K[])
{
  int i;
  for (i=0; i<N; i++)
    {
      K[i]=0.5*(pow(V[i],2) + pow(V[i+N],2) + pow(V[i+2*N],2));
    }
}


