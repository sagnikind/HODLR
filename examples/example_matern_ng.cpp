#include "HODLR_Matrix.hpp"
#include "HODLR.hpp"
//#include <gsl/gsl_sf_bessel.h>
//#include <gsl/gsl_errno.h>
//#include <gsl/gsl_fft_complex.h>

/** ****************************************************************************

  Structure for identifies two dimensions struct with two vectors (x and y).
 **/
typedef struct {
	double *x; ///< Values in X dimension.
	double *y; ///< Values in Y dimension.
	double *z; ///< Values in Z dimension.
} location;

double uniform_distribution(double rangeLow, double rangeHigh) 
	//! Generate uniform distribution between rangeLow , rangeHigh
{
	// unsigned int *seed = &exageostat_seed;
	double myRand = (double) rand() / (double) (1.0 + RAND_MAX);
	double range = rangeHigh - rangeLow;
	double myRand_scaled = (myRand * range) + rangeLow;
	return myRand_scaled;
}

static uint32_t Part1By1(uint32_t x)
// Spread lower bits of input
{
  x &= 0x0000ffff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555;
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

static uint32_t EncodeMorton2(uint32_t x, uint32_t y)
// Encode two inputs into one
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint32_t Compact1By1(uint32_t x)
// Collect every second bit into lower part of input
{
  x &= 0x55555555;
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

static uint32_t DecodeMorton2X(uint32_t code)
// Decode first input
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
// Decode second input
{
    return Compact1By1(code >> 1);
}

static int compare_uint32(const void *a, const void *b)
    //! Compare two uint32_t
{
    uint32_t _a = *(uint32_t *)a;
    uint32_t _b = *(uint32_t *)b;
    if(_a < _b) return -1;
    if(_a == _b) return 0;
    return 1;
}

static void zsort_locations(int n, location * locations)
//! Sort in Morton order (input points must be in [0;1]x[0;1] square])
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    // Encode data into vector z
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(locations->x[i]*(double)UINT16_MAX +.5);
        y = (uint16_t)(locations->y[i]*(double)UINT16_MAX +.5);
        //printf("%f %f -> %u %u\n", points[i], points[i+n], x, y);
        z[i] = EncodeMorton2(x, y);
    }
    // Sort vector z
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    // Decode data from vector z
    for(i = 0; i < n; i++)
    {
        x = DecodeMorton2X(z[i]);
        y = DecodeMorton2Y(z[i]);
        locations->x[i] = (double)x/(double)UINT16_MAX;
        locations->y[i] = (double)y/(double)UINT16_MAX;
        //printf("%lu (%u %u) -> %f %f\n", z[i], x, y, points[i], points[i+n]);
    }
}

location* GenerateXYLoc(int n, int seed)
	//! Generate XY location for exact computation (MORSE)
{
	//initalization
	int i = 0 ,index = 0, j = 0;
	// unsigned int *seed = &exageostat_seed;
	srand(seed);
	location* locations = (location *) malloc( sizeof(location*));
	//Allocate memory
	locations->x        = (double *) malloc(n * sizeof(double));
	locations->y        = (double *) malloc(n * sizeof(double));
	locations->z		= NULL;
	// if(strcmp(locs_file, "") == 0)
	// {

	int sqrtn = ceil(sqrt(n));

	//Check if the input is square number or not
	//if(pow(sqrtn,2) != n)
	//{
	//printf("n=%d, Please use a perfect square number to generate a valid synthetic dataset.....\n\n", n);
	//exit(0);
	//}

	int *grid = (int *) calloc((int)sqrtn, sizeof(int));

	for(i = 0; i < sqrtn; i++)
	{
		grid[i] = i+1;
	}

	for(i = 0; i < sqrtn && index < n; i++)
		for(j = 0; j < sqrtn && index < n; j++){
			locations->x[index] = (grid[i]-0.5+uniform_distribution(-0.4, 0.4))/sqrtn;
			locations->y[index] = (grid[j]-0.5+uniform_distribution(-0.4, 0.4))/sqrtn;
			//printf("%f, %f\n", locations->x[index], locations->y[index]);
			index++;
		}
	free(grid);
	zsort_locations(n, locations);
	return locations;
}

class Kernel : public HODLR_Matrix 
{
	private:
		location* x;
		double phi, nu;

	public:

		// Constructor:
		Kernel(int N, double phi, double nu, int seed) : HODLR_Matrix(N) 
	{
		x =  GenerateXYLoc( N,  seed);

		this->phi = phi;
		this->nu  = nu;
		// This is being sorted to ensure that we get
		// optimal low rank structure:
		zsort_locations(N, x);
	};

		dtype getMatrixEntry(int i, int j) 
		{
			double con = 0.0;
			double dist = 0.0;
			//dist = 4 * sqrt(2*nu) * sqrt(pow((x->x[i] - x->x[j]), 2) + pow((x->y[i] - x->y[j]), 2))/phi;
			dist = 4  * sqrt(pow((x->x[i] - x->x[j]), 2) + pow((x->y[i] - x->y[j]), 2))/phi;

			//con = pow(2,(nu-1)) * tgamma(nu);
			//con = 1.0/con;

			if(dist == 0)
				return 1;
			else
				return exp(-dist); //con*pow(dist, nu) ;//* gsl_sf_bessel_Knu(nu, dist);
		}

		// Destructor:
		~Kernel() {};
};

void core_g_to_ng (double *Z, double *localtheta, int m) {

	//localtheta[2] ---> \xi (location)
	//localtheta[3] ---> \omega (scale)
	//localtheta[4] ---> g (skewness of tukey_gh)
	//localtheta[5] ---> h (kurtosis of tukey_gh)

	double xi    = localtheta[2];
	double omega = localtheta[3];
	double g     = localtheta[4];
	double h     = localtheta[5];


	int i;
	if(h<0)
	{
		printf("The kurtosis parameter cannot be negative");
		return;
	}
	if(g == 0)
	{
		for(i = 0; i < m; i++)
			Z[i] = xi + omega *  Z[i] * (exp(0.5 * h * pow(Z[i], 2)));
	}
	else
	{
		for(i = 0; i < m; i++)
			Z[i] = xi + omega * (exp(g * Z[i]) - 1) * (exp(0.5 * h * pow(Z[i], 2))) / g;
	}

}

double Tukey_gh(double *theta, int N)
        //! Generate from Tukey_gh 
{
	int i;
        Vec z;
	double PI = 3.14;
	for(i=0;i<N;i++)
	{
		z(i) = sqrt(-2*log(uniform_distribution(0,1))) * cos(2*PI*uniform_distribution(0,1));
	}

	Kernel* K            = new Kernel(N, theta[0], theta[1], 0);
	
	Mat B = K->getMatrix(0, 0, N, N);
        
	Eigen::LLT<Mat> llt; 
	Mat L = llt.matrixL();
	
	z = L * z ;

	double *z_0[N];
	for(i=0;i<N;i++)
	{	
		*z_0[i] = z(i);
	}

	//core_g_to_ng(z_0,theta,N);
	
	//return z_0;
} 

static double f(double z_non, double z, double xi, double omega, double g, double h)
{
	if(g == 0)
		return z_non - xi - omega* z*exp(0.5*h*z*z);
	else

		return z_non- xi - (omega * (exp(g * z) - 1) * (exp(0.5 * h * z*z)) / g);
}
static double df(double z, double xi, double omega, double g, double h)
{
	if(g == 0)
		return - omega*exp((h*z*z)/2.0) - omega*h*z*z*exp((h*z*z)/2.0);
	else
		return - omega*exp(g*z)*exp((h*z*z)/2.0) - (h*z*exp((h*z*z)/2.0)*(omega*exp(g*z) - omega))/g;
}
double newton_raphson(double z, double xi, double omega, double g, double h, double eps)
{
	int itr, maxmitr;
	double x0, x1, allerr;
	x0 = 0;
	double diff;
	allerr = eps;
	maxmitr = 1000;
	for (itr=1; itr<=maxmitr; itr++)
	{
		diff = f(z, x0, xi, omega, g, h)/df(x0, xi, omega, g, h);
		x1 = x0-diff;
		//if(isnan(x1))
		//	return x0;
		if (fabs(diff) < allerr)
			return x1;
		x0 = x1;
	}

	return x1;
}

void core_ng_transform (double *Z, double *nan_flag, double *localtheta, int m) {

	//localtheta[2] ---> \xi (location)
	//localtheta[3] ---> \omega (scale)
	//localtheta[4] ---> g (skewness of tukey_gh)
	//localtheta[5] ---> h (kurtosis of tukey_gh)
	double xi    = localtheta[2];
	double omega = localtheta[3];
	double g     = localtheta[4];
	double h     = localtheta[5];

	double eps = 1.0e-5;
	int i=0;
	for(i = 0; i < m; i++)
		Z[i] =  newton_raphson(Z[i], xi, omega, g, h, eps);
}

double core_ng_loglike (double *Z, double *localtheta, int m) {

    //localtheta[2] ---> \xi (location)
    //localtheta[3] ---> \omega (scale)
    //localtheta[4] ---> g (skewness of tukey_gh)
    //localtheta[5] ---> h (kurtosis of tukey_gh)

    double xi    = localtheta[2];
    double omega = localtheta[3];
    double g     = localtheta[4];
    double h     = localtheta[5];

    int i;
    double sum = 0;
    if(h < 0)
    {
        printf("kurtosis parameter cannot be negative");
        exit(1);
    }
    for(i = 0; i < m; i++)
    {
        if(g == 0)
            sum += log(1 + h * pow(Z[i], 2)) + 0.5 * h * pow(Z[i], 2);
        else
        {
            sum += log(exp(g * Z[i]) + (exp(g * Z[i])-1) * h * Z[i]/g ) + 0.5 * h * pow(Z[i],2);
            //printf("g:%f, h:%f, Z[i]:%f,", g, h, Z[i]);
        }
    }
    return(sum);
}

double MLE_ng_dense(double *z, int N, double *theta)
{	
	int i;
	double PI = 3.14;
	double *nan_flag = 0;

	Kernel* K            = new Kernel(N, theta[0], theta[1], 0);
	
	core_ng_transform(z,nan_flag ,theta,N);

	Vec z_0;
        for(i=0;i<N;i++)
        {
                z_0(i) = z[i];
        }

        Mat B = K->getMatrix(0, 0, N, N);
	
	Eigen::LLT<Mat> llt;
	Vec x = B.llt().solve(z_0);
	
	double dotp = z_0.adjoint()*x;
	double log_det = log(B.determinant());
	double loglik = -0.5 * dotp -  0.5*log_det ;
        loglik = loglik - core_ng_loglike (z, theta, N) - N * log(theta[3]) - (double) (N / 2.0) * log(2.0 * PI);
        return loglik;

}

/*double MLE_ng_HODLLR(double *z,int n, double *theta, int N, int M, int tol)
{
	int i;
	tol = pow(10, - tol);
	Kernel* K            = new Kernel(N, theta[0], theta[1], 0);
	HODLR* T = new HODLR(N, M, tol);
	T->assemble(K, "rookPivoting", 1, 1);
	core_ng_transform(z,nan_flag ,theta,N);
	Mat x;
        x = T->solve(z);
	double dotp = 0;
	for(i=0;i<n;i++)
	{
		dotp += x(i,0)*z[i];
	}

	dtype log_det_hodlr = T->logDeterminant();
	double loglik = -0.5 * dotp -  0.5*log_det_hodlr ;
	loglik = loglik - core_ng_loglike (z, theta, n) - N * log(theta[3]) - (double) (N / 2.0) * log(2.0 * PI);
	return loglik;
}*/

int main(int argc, char* argv[]) 
{
	int N, M;
	double tolerance;
        double phi,nu;

	if(argc < 6)
	{
		std::cout << "All arguments weren't passed to executable!" << std::endl;
		std::cout << "Using Default Arguments:" << std::endl;
		// Size of the Matrix in consideration:
		N          = 10000;
		// Size of Matrices at leaf level:
		M          = 200;
		// Tolerance of problem
		tolerance  = pow(10, -12);
		phi = 0.1;
		nu = 0.5;
	}

	else
	{
		// Size of the Matrix in consideration:
		N          = atoi(argv[1]);
		// Size of Matrices at leaf level:
		M          = atoi(argv[2]);
		// Tolerance of problem
		tolerance  = pow(10, -atoi(argv[3]));
		phi = atof(argv[4]);
		nu = atof(argv[5]);
	}

	// Declaration of HODLR_Matrix object that abstracts data in Matrix:
	Kernel* K            = new Kernel(N, phi, nu, 0);

	std::cout << "========================= Problem Parameters =========================" << std::endl;
	std::cout << "Matrix Size                        :" << N << std::endl;
	std::cout << "Leaf Size                          :" << M << std::endl;
	std::cout << "Tolerance                          :" << tolerance << std::endl << std::endl;

	// Variables used in timing:
	double start, end;

	std::cout << "Fast method..." << std::endl;

	bool is_sym = true;
	bool is_pd  = true;

	// Creating a pointer to the HODLR Tree structure:
	HODLR* T = new HODLR(N, M, tolerance);

	start = omp_get_wtime();
	T->assemble(K, "rookPivoting", is_sym, is_pd);
	end = omp_get_wtime();
	std::cout << "Time for assembly in HODLR form:" << (end - start) << std::endl;

	// Random Matrix to multiply with
	Mat x = (Mat::Random(N, 1)).real();
	// Stores the result after multiplication:
	Mat y_fast, b_fast;

	start  = omp_get_wtime();
	b_fast = T->matmatProduct(x);
	end    = omp_get_wtime();

	std::cout << "Time for matrix-vector product:" << (end - start) << std::endl << std::endl;

	start = omp_get_wtime();
	T->factorize();
	end   = omp_get_wtime();
	std::cout << "Time to factorize:" << (end-start) << std::endl;

	Mat x_fast;
	start  = omp_get_wtime();
	x_fast = T->solve(b_fast);
	end    = omp_get_wtime();

	std::cout << "Time to solve:" << (end-start) << std::endl;

	if(is_sym == true && is_pd == true)
	{
		start  = omp_get_wtime();
		y_fast = T->symmetricFactorTransposeProduct(x);
		end    = omp_get_wtime();
		std::cout << "Time to calculate product of factor transpose with given vector:" << (end - start) << std::endl;

		start  = omp_get_wtime();
		b_fast = T->symmetricFactorProduct(y_fast);
		end    = omp_get_wtime();
		std::cout << "Time to calculate product of factor with given vector:" << (end - start) << std::endl;        
	}

	start = omp_get_wtime();
	dtype log_det_hodlr = T->logDeterminant();
	end = omp_get_wtime();
	std::cout << "Time to calculate log determinant using HODLR:" << (end-start) << std::endl;

	// Direct method:
	start = omp_get_wtime();
	Mat B = K->getMatrix(0, 0, N, N);
	end   = omp_get_wtime();

	if(is_sym == true && is_pd == true)
	{
		start = omp_get_wtime();
		Eigen::LLT<Mat> llt;
		llt.compute(B);
		end = omp_get_wtime();
		std::cout << "Time to calculate LLT Factorization:" << (end-start) << std::endl;
	}

	else
	{
		start = omp_get_wtime();
		Eigen::PartialPivLU<Mat> lu;
		lu.compute(B);
		end = omp_get_wtime();
		std::cout << "Time to calculate LU Factorization:" << (end-start) << std::endl;        
	}

	delete K;
	delete T;

	return 0;
}
