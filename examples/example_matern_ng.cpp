#include "HODLR_Matrix.hpp"
#include "HODLR.hpp"
#include <math.h>
#include <sys/stat.h>
#include <nlopt.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <lapacke.h>
/** ****************************************************************************

  Structure for identifies two dimensions struct with two vectors (x and y).
 **/
int count = 0;
int iters = 0;
double nan_flag = 0;
typedef struct {
    double *x; ///< Values in X dimension.
    double *y; ///< Values in Y dimension.
    double *z; ///< Values in Z dimension.
} location;

typedef struct
{
    double *obs; //observations.
    double *initial_theta;
    double *starting_theta;
    int N;       //Matrix size.
    int nLeaf;   //HODLR nLeaf parameter.
    double tol;     //HODLR tol parameter.
    int seed;
} MLE_HODLR_data;

typedef struct {
    double final_loglik;
    double time_per_iteration;
    double time_per_generation;
    double time_per_factorization;
    double time_per_solver;
    double *initial_theta;
    double *estimated_theta;
} output;


output results;

static double calculateDistance( location* l1, location* l2, int l1_index,
        int l2_index, int distance_metric, int z_flag) {

    double z1, z2;
    double x1=l1->x[l1_index];
    double y1=l1->y[l1_index];
    double x2=l2->x[l2_index];
    double y2=l2->y[l2_index];
    if(l1->z == NULL || l2->z == NULL || z_flag == 1)
    {
        if(distance_metric == 1)
            std::cout<<"Great Circle (GC) distance is not supported here"<<"\n";
        return  sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
    }
    else
    {
        if(distance_metric == 1)
        {
            printf("Great Circle (GC) distance is only valid for 2d\n");
            exit(0);
        }
        z1 = l1->z[l1_index];
        z2 = l2->z[l2_index];
        return  sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2) +  pow((z2 - z1), 2));
    }
}

void init_optimizer( nlopt_opt * opt, double *lb, double *ub, double tol)
    //! Initialize the NLOPT optimizer
    /*!
     * Returns nlopt_opt object.
     * @param[in] lb: optimization lower bounds vector ( lb_1, lb_2, lb_3).
     * @param[in] up: optimization upper bounds vector ( ub_1, ub_2, ub_3).
     * @param[in] tol: a tolerance that is used for the purpose of stopping criteria only.
     * @param[out] opt: nlopt_opt object.
     * */
{
    //initalizing opt library
    nlopt_set_lower_bounds(*opt, lb);
    nlopt_set_upper_bounds(*opt, ub);
    nlopt_set_ftol_abs(*opt, tol);
}


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

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

void write_vectors(double * zvec, location * locations, int n)
    //! store locations, measurements, and log files if log=1
    /*!
     * Returns initial_theta, starting_theta, target_theta.
     * @param[in] zvec: measurements vector.
     * @param[in] data: MLE_data struct with different MLE inputs.
     * @param[in] n: number of spatial locations
     * */
{

    int i = 1;
    FILE *pFileZ, *pFileXY;
    location *l = locations;
    struct stat st = {0};
    char * nFileZ  = (char *) malloc(50 * sizeof(char));
    char * temp    = (char *) malloc(50 * sizeof(char));
    char * nFileXY = (char *) malloc(50 * sizeof(char));
    char * nFileLog = (char *) malloc(50 * sizeof(char));
    //Create New directory if not exist
    if (stat("./synthetic_ds", &st) == -1)
        mkdir("./synthetic_ds", 0700);

    snprintf(nFileZ, 50, "%s%d%s", "./synthetic_ds/Z_", n,"_");
    snprintf(nFileXY, 50, "%s%d%s", "./synthetic_ds/LOC_", n,"_");
    snprintf(nFileLog, 50, "%s%d%s", "./synthetic_ds/log_", n,"_");

    snprintf(temp, 50, "%s%d", nFileLog , i);
    while(is_file_exist(temp) == 1)
    {
        i++;
        snprintf(temp, 50, "%s%d", nFileLog , i);
    }

    sprintf(temp, "%d", i);
    strcat(nFileZ, temp);
    strcat(nFileXY, temp);
    strcat(nFileLog, temp);


    pFileZ = fopen(nFileZ, "w+");
    pFileXY = fopen(nFileXY, "w+");


    for(i=0;i<n;i++){
        fprintf(pFileZ, "%0.12f\n", zvec[i]);
        //if(l->z ==NULL)
        fprintf(pFileXY, "%0.12f,%0.12f\n", l->x[i], l->y[i]);
        //else
        //  fprintf(pFileXY, "%f,%f,%f\n", l->x[i], l->y[i], l->z[i]);
    }

    fclose(pFileZ);
    fclose(pFileXY);
}

class Kernel : public HODLR_Matrix 
{
    private:
        location* l;
        double phi, nu;

    public:
        Mat A;
        // Constructor:
        Kernel(int N, double phi, double nu, int seed) : HODLR_Matrix(N) 
    {       
        l =  GenerateXYLoc( N,  seed);

        this->phi = phi;
        this->nu  = nu;
        // This is being sorted to ensure that we get
        // optimal low rank structure:
        zsort_locations(N, l);
    };

        dtype getMatrixEntry(int i, int j) 
        {
            double sigma_square = 1;
            double con = 0.0;
            double dist = 0.0;
            con = pow(2,(nu-1)) * tgamma(nu);
            con = 1.0/con;
            con = sigma_square * con;
            dist = 4 * sqrt(2*nu) * (calculateDistance(l, l, i, j, 0, 0) / phi);
            if(i == j)
                return sigma_square;
            // Otherwise:
            else
                return con*pow(dist, nu)	* gsl_sf_bessel_Knu(nu, dist);
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

double *Tukey_gh(double *theta, int N,int seed, MLE_HODLR_data* data)
    //! Generate from Tukey_gh 
{
    srand(seed);
    int i;
    double PI = 3.141592653589793238;
    double *z_0=(double *) malloc(N * sizeof(double)) ;

    Eigen::VectorXd z(N);

    //std::cout << "N=   " << N << std::endl;

    //for(i=0;i<N;i++)
    //{	
    //	z[i] = sqrt(-2*log(uniform_distribution(0,1))) * cos(2*PI*uniform_distribution(0,1));
    //	}
    int iseed[4]={seed, seed, seed, 1};
    LAPACKE_dlarnv(3, iseed, N, z_0);

    Kernel* K            = new Kernel(N, theta[0], theta[1], 0);

    //	bool is_sym = true;
    //	bool is_pd  = true;
    //	HODLR* T = new HODLR(N, data->nLeaf, data->tol);
    //	T->assemble(K, "rookPivoting", is_sym, is_pd);
    Mat B = K->getMatrix(0, 0, N, N);
    //std::cout << "Covariance matrix:\n" << B << std::endl;

    Eigen::LLT<Mat> llt;
    llt.compute(B);	
    Mat L = llt.matrixL();
    for(i=0;i<N;i++)
        z[i] = z_0[i];
    //std::cout << "L Covariance matrix:\n" << L << std::endl;
    z = L * z ;

    for(i=0;i<N;i++)
        z_0[i] = z[i];
    //std::cout << "Normal obs:\n" << z << std::endl;

    core_g_to_ng(z_0,theta,N);

    //std::cout << "TGH obs:\n" << z << std::endl;
    return z_0;
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

void core_ng_transform (double *Z,const double *localtheta, int m) {

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

    //exit(0);	
    Eigen::VectorXd z(m);

    for(i=0;i<m;i++)
    {
        z[i] = Z[i];
    }

    // std::cout << "After transformation:\n" << z << std::endl;
}

double core_ng_loglike (double *Z,const double *localtheta, int m) {

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

double MLE_ng_dense(double *z_0, int N, double *theta)
{	
    int i;
    double PI = 3.141592653589793238;
    //double FLT_MAX = pow(10,7);
    double nan_flag = 0;
    double start, end;

    Kernel* K            = new Kernel(N, theta[0], theta[1], 0);
    //Mat B = K->getMatrix(0, 0, N, N);
    //std::cout << "Covariance matrix:\n" << B << std::endl;

    start = omp_get_wtime();
    core_ng_transform(z_0,theta,N);
    end = omp_get_wtime();
    std::cout << "Time for core_ng_transform in dense: " << (end - start) << std::endl;
    for (i=0;i<N;i++)
        if(isnan(z_0[i]))
            nan_flag = 1;
    if(nan_flag == 1)
    {	
        printf("Inf case\n");
        return -FLT_MAX;	
    }
    else
    {
        Eigen::VectorXd z(N);
        for(i=0;i<N;i++)
        {
            z[i] = z_0[i];
        }		
        Mat B = K->getMatrix(0, 0, N, N);

        start = omp_get_wtime();
        Eigen::LLT<Mat> llt;
        Vec x = B.llt().solve(z);

        double dotp = z.adjoint()*x;
        end = omp_get_wtime();
        std::cout << "Time for dotp in dense: " << (end - start) << std::endl;

        start = omp_get_wtime();
        llt.compute(B);
        dtype log_det = 0.0;
        for(int i = 0; i < llt.matrixL().rows(); i++)
        {
            log_det += log(llt.matrixL()(i,i));
        }
        log_det *= 2;
        end = omp_get_wtime();
        std::cout << "Time for logdet in dense: " << (end - start) << std::endl;	

        double loglik = -0.5 * dotp -  0.5*log_det ;
        loglik = loglik - core_ng_loglike (z_0, theta, N) - N * log(theta[3]) - (double) (N / 2.0) * log(2.0 * PI);
        // std::cout << "dotp:" << std::setprecision(16) <<dotp << std::endl;
        // std::cout << "logdet:" << std::setprecision(16) <<log_det << std::endl;
        // std::cout << "core_ng_loglike:" << std::setprecision(16) << core_ng_loglike (z_0, theta, N) << std::endl;
        // std::cout << "logtheta:" <<std::setprecision(16) << N * log(theta[3]) << std::endl;
        // std::cout << "loglik:" <<std::setprecision(16) << loglik << std::endl;
        return loglik;
    }
}

double* dense_data_generation_non_gaussian(MLE_HODLR_data *data)
{

    double *z=(double *) malloc(data->N * sizeof(double)) ;

    z = Tukey_gh(data->initial_theta, data->N, data->seed, data);

    return z;
}

double MLE_ng_HODLR(double *z_0, const double *theta, int N, int M, double tol)
{
    int i;
    double PI = 3.141592653589793238;
    //double FLT_MAX = pow(10,7);
    nan_flag = 0;
    double tol_print =tol;
    tol = pow(10, tol);
    double start, end;
    bool is_sym = true;
    bool is_pd  = true;
    double *z_temp=(double *) malloc(N * sizeof(double)) ;

    for (i=0; i<N; i++)
        z_temp[i] = z_0[i];

    start = omp_get_wtime();
    core_ng_transform(z_temp, theta, N);
    end = omp_get_wtime();

    std::cout << "Time for core_ng_transform in HODLR: " << (end - start) << std::endl;

    for (i=0;i<N;i++)
        if(__isnan(z_temp[i]))
            nan_flag = 1;

    if(nan_flag == 1)
    {
        std::cout <<iters++<<"- ("<<theta[0]<<", "<<theta[1]<<", "<< theta[2]<<", "<<theta[3]<<", "<< theta[4]<<", "<<theta[5]<<")---> LogLi (inf case) = "<<  -FLT_MAX<<"\n";
        std::cout << "========================================================= "<< std::endl;
        return -FLT_MAX;
    }
    else
    {
        Eigen::VectorXd z(N);
        for(i=0;i<N;i++)
        {
            z[i] = z_temp[i];
        }

        Kernel* K            = new Kernel(N, theta[0], theta[1], 0);
        //printf("%d, %d, %e\n", N, M ,tol);
        HODLR* T = new HODLR(N, M, tol);
        start = omp_get_wtime();
        //  std::cout <<count++<<"- ("<<theta[0]<<", "<<theta[1]<<", "<< theta[2]<<", "<<theta[3]<<", "<< theta[4]<<", "<<theta[5]<<")\n";
        std::cout << "Making cov mat in HODLR form ... " << std::endl;
        T->assemble(K, "rookPivoting", is_sym, is_pd);
        //std::string path = "image"+std::to_string(theta[0])+"-"+std::to_string(theta[1])+"-"+std::to_string(theta[2])+"-"+std::to_string(theta[3])+"-"+std::to_string(theta[4])+"-"+std::to_string(theta[5])+"-"+ std::to_string(M) + "-"+std::to_string(tol_print)+".svg";
        //T->plotTree(path);
        //T->printTreeDetails();
        end = omp_get_wtime();
        std::cout << "Time for making cov mat in HODLR form: " << (end - start) << std::endl;
        results.time_per_iteration +=(end - start);
        results.time_per_generation +=(end - start);

        //Factorization 
        start = omp_get_wtime();
        Eigen::LLT<Mat> llt;
        T->factorize();
        end = omp_get_wtime();
        std::cout << "Time for making cov mat in HODLR form: " << (end - start) << std::endl;
        results.time_per_iteration +=(end - start);
        results.time_per_factorization +=(end - start);

        //Solver
        start = omp_get_wtime();
        Vec x = T->solve(z);
        end = omp_get_wtime();
        std::cout << "Time for solver trsm in HODLR form: " << (end - start) << std::endl;
        results.time_per_iteration +=(end - start);
        results.time_per_solver +=(end - start);
        for(i=0;i<N;i++)
            z_temp[i] = z[i];
        double dotp = z.adjoint()*x;

        //exit(0);  to draw heatmap
        std::cout << "Time to factorize in HODLR: " << (end - start) << std::endl;

        start = omp_get_wtime();
        dtype log_det_hodlr = T->logDeterminant();
        end = omp_get_wtime();
        std::cout << "Time for logdet in HODLR: " << (end - start) << std::endl;
        results.time_per_iteration +=(end - start);
        double loglik = -0.5 * dotp -  0.5*log_det_hodlr;

        loglik = loglik - core_ng_loglike (z_temp, theta, N) - N * log(theta[3]) - (double) ( N / 2.0) * log(2.0 * PI);

        std::cout <<iters++<<"- ("<<theta[0]<<", "<<theta[1]<<", "<< theta[2]<<", "<<theta[3]<<", "<< theta[4]<<", "<<theta[5]<<")---> LogLi = "<< std::setprecision(16) << loglik<<"\n";
        count++;
        std::cout << " ---- dotp: " << std::setprecision(16) <<dotp << std::endl;
        std::cout << " ---- logdet: " << std::setprecision(16)  << log_det_hodlr << std::endl;
        std::cout << " ---- core_ng_loglike: " <<std::setprecision(16) << core_ng_loglike (z_temp, theta, N) << std::endl;
        //std::cout << "logtheta:" <<std::setprecision(16) << N * log(theta[3]) << std::endl;
        std::cout << " ---- loglik: " << std::setprecision(16) <<loglik << std::endl;
        std::cout << "========================================================= "<< std::endl;
        results.final_loglik = loglik;
        for(int i=0;i<6;i++)
            results.estimated_theta[i] = theta[i];
        return loglik;
    }
}


void HODLR_non_gaussian( int argc, char* argv[])
{
    double *theta=(double *) malloc(6 * sizeof(double)) ;
    double *initial_theta=(double *) malloc(6 * sizeof(double)) ;
    int N,M,tol;

    if(argc < 15)
    {
        std::cout << "All arguments weren't passed to executable!" << std::endl;
        std::cout << "Using Default Arguments:" << std::endl;

        theta[0] = 10;
        theta[1] = 0.7;
        theta[2] = 5;
        theta[3] = 1;
        theta[4] = 0.2;
        theta[5] = 0.2;

        initial_theta[0] = 0.1;
        initial_theta[1] = 0.5;
        initial_theta[2] = 0;
        initial_theta[3] = 2;
        initial_theta[4] = 0.1;
        initial_theta[5] = 0.1;
        N = 10000;
        M = 100;
        tol = 12;
    }

    else
    {	
        theta[0] = atof(argv[1]);
        theta[1] = atof(argv[2]);
        theta[2] = atof(argv[3]);
        theta[3] = atof(argv[4]);
        theta[4] = atof(argv[5]);
        theta[5] = atof(argv[6]);

        initial_theta[0] = atof(argv[7]);
        initial_theta[1] = atof(argv[8]);
        initial_theta[2] = atof(argv[9]);
        initial_theta[3] = atof(argv[10]);
        initial_theta[4] = atof(argv[11]);
        initial_theta[5] = atof(argv[12]);
        N = atoi(argv[13]);
        M = atoi(argv[14]);
        tol = atoi(argv[15]);
    }

    double *z=(double *) malloc(N * sizeof(double)) ;
    //z = Tukey_gh(theta, N,0, data);

    //Eigen::VectorXd Z(N);
    //int i;
    //for(i=0;i<N;i++)
    //{
    //        Z[i] = z[i];
    //}

    //std::cout << "hodlr z:" << Z << std::endl;
    //location* x;
    //int seed = 14;//remove after testing
    //x =  GenerateXYLoc( N,  seed);
    //write_vectors(z,x,N);


    MLE_ng_HODLR( z, initial_theta, N, M, tol);
}


double MLE_ng_HODLR_alg(unsigned n, const double * theta, double * grad, void * data)
{
    double *z;// = (double *) malloc(n * sizeof(double)) ;
    z = ((MLE_HODLR_data*)data)->obs;
    int N = ((MLE_HODLR_data*)data)->N;
    int nLeaf = ((MLE_HODLR_data*)data)->nLeaf;
    double tol = ((MLE_HODLR_data*)data)->tol;
    return MLE_ng_HODLR(z, theta, N, nLeaf, tol);
}

int main(int argc, char* argv[])
{	
    gsl_set_error_handler_off () ;	
    //HODLR_non_gaussian(argc, argv);
    double  opt_f;
    nlopt_opt opt;
    MLE_HODLR_data data;
    int num_params = 6;
    int j;

    if(argc < 15)
    {
        std::cout << "All arguments weren't passed to executable!" << std::endl;
        //std::cout << "Using Default Arguments:" << std::endl;
        exit(0);
    }

    double *starting_theta  = (double *) malloc(num_params * sizeof(double));
    double *initial_theta   = (double *) malloc(num_params * sizeof(double));
    results.estimated_theta  = (double *) malloc(num_params * sizeof(double));
    results.initial_theta   = (double *) malloc(num_params * sizeof(double));

    results.final_loglik = 0;
    results.time_per_generation = 0;
    results.time_per_factorization = 0;
    results.time_per_solver = 0;
    results.time_per_iteration = 0;


    double* lb = (double *) malloc(num_params * sizeof(double));
    double* ub = (double *) malloc(num_params * sizeof(double));


    int N           = atoi(argv[1]);
    int nLeaf       = atoi(argv[2]);
    double tol      = atof(argv[3]);
    double opt_tol  = atof(argv[4]);

    //lb[0] = 0.01; lb[1] = 0.01; lb[2] = -5; lb[3] = 0.01;  lb[4] = -2;  lb[5] = 0.01;
    //ub[0] = 10;  ub[1] = 5;  ub[2] = 5;   ub[3] = 5;  ub[4] = 2;   ub[5] = 1;

    for(int i = 0; i<num_params; i++)
    {
        lb[i]               = atof(argv[i+5]);
        ub[i]               = atof(argv[i+11]);
        initial_theta[i]    = atof(argv[i+17]);
        results.initial_theta[i] = initial_theta[i];
        starting_theta[i]   = lb[i];
        //        std::cout <<lb[i]<<"  "<<ub[i]<<"  "<<initial_theta[i]<<std::endl;
    }
    int seed =  atoi(argv[23]);
    int opt_iters =  atoi(argv[24]);    
    data.seed= seed;
    starting_theta[4]=0.2;
    starting_theta[5]=0.2;

    //set data struct values
    data.N       = N;
    data.nLeaf   = nLeaf;
    data.tol     = tol;
    data.initial_theta  = initial_theta;
    data.starting_theta = starting_theta;    

    std::cout << "========================= Problem Parameters =========================" << std::endl;
    std::cout << "Matrix Size                        :" << data.N << std::endl;
    std::cout << "Leaf Size                          :" << data.nLeaf << std::endl;
    std::cout << "HODLR Tolerance                    :" << data.tol << std::endl;
    std::cout << "Seed(z_sample)                     :" << data.seed << std::endl;
    std::cout << "Opt. Iterations                    :" << opt_iters << std::endl << std::endl;

    //Generate new dataset
    std::cout << "Synthetic data generatation in dense format... "<<std::endl;
    double start = omp_get_wtime();
    data.obs = dense_data_generation_non_gaussian(&data);
    std::cout << "Done \n ";
    double end = omp_get_wtime();
    std::cout << "Time : " << (end - start) << std::endl;

    //    starting_theta[0]=0.96;  starting_theta[1]=0.5;  starting_theta[2]=0;  starting_theta[3]=2;  starting_theta[4]=0.5;  starting_theta[5]=0.3;

    opt=nlopt_create(NLOPT_LN_BOBYQA, num_params);
    init_optimizer(&opt, lb, ub, opt_tol);
    nlopt_set_maxeval(opt, opt_iters);

    nlopt_set_max_objective(opt, MLE_ng_HODLR_alg, &data);
    nlopt_optimize(opt, starting_theta, &opt_f);
    //****************
    FILE *pFile;
    char * nFileZ  = (char *) malloc(50 * sizeof(char));


    snprintf(nFileZ, 50, "%s%f%s%f%s%f%s", "./results_hodlr_", initial_theta[0],"_",  initial_theta[4],"_", initial_theta[5],".log");

    pFile = fopen(nFileZ, "a");

    if(pFile == NULL) {
        printf("Cannot access the results path(3)\n");
        return -1;
    }

    fprintf(pFile, "%d ", data.N);
    fprintf(pFile, "%d ", data.nLeaf);
    fprintf(pFile, "%f ", data.tol);
    fprintf(pFile, "%d ", seed);   //z_sample

    for(int i=0; i<6; i++)
        fprintf(pFile, "%6.6f ", results.initial_theta[i]);
    for(int i=0; i<6; i++)
        fprintf(pFile, "%6.6f ", results.estimated_theta[i]);

    fprintf(pFile, "%6.6f ", results.final_loglik);
    fprintf(pFile, "%d ", iters);
    fprintf(pFile, "%6.6f ", results.time_per_generation/count);
    fprintf(pFile, "%6.6f ", results.time_per_factorization/count);
    fprintf(pFile, "%6.6f ", results.time_per_solver/count);
    fprintf(pFile, "%6.6f\n", results.time_per_iteration/count);

    fclose(pFile);
}

/*int main(int argc, char* argv[]) 
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
}*/
