// DP-BSS Demo

#define  ARMA_USE_LAPACK
#include <iostream>
#include <fstream>
#include <string>
#include <armadillo>
#include <random>
#include <vector>
#include <numeric>
#include <ctime>

// For path and directory
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace arma;

class DP_BSS {
    public:
        // Initialization
        void init(int sparse, int eps, double sens_scale, int max_iteration,
                    bool standardization, string init, int F1_size) {
            s = sparse;
            epsilon = eps;
            sensitivity_scaling = sens_scale;
            max_iter = max_iteration;
            standardize = standardization;
            initialization = init;
            F1_score_size = F1_size;
        };

        // other parameter
        Row<int> S;
        vector<Row<int>> S_list; // store last # F1_score_size S
        vector<double> log_pi_n_list;
        vector<double> RSS_list;

        // Functions
        void fit(arma::Mat<double>& X_in, arma::Col<double>& y_in) {
            n = X_in.n_rows;
            p = X_in.n_cols;
            // standardize X and y
            if (standardize) {
                X = X_in;
                rowvec colmean = mean(X_in, 0);
                for (uword i=1; i<X.n_cols; ++i) {
                    X.col(i) -= colmean(i);
                }
                y = y_in - as_scalar(mean(y_in));
            };
            MCMC();
        };

    
    private:
        // tuning parameters
        int s = 5;
        int epsilon = 5;
        double sensitivity_scaling = 20.0;

        // MCMC parameters
        int max_iter = 2000;

        // some options
        bool standardize = true;
        string initialization = "Random";

        // data
        int n; // row size
        int p; // col size
        mat X; // Mat<double>, dim = (n, p)
        vec y; // Col<double>, dim = n

        // None 0 beta term
        Row<int> initial_state; // dim = p

        // other parameters
        mat Y_S_list; // each column is Y_S
        int acceptance;
        double RSS;
        int F1_score_size;

        // helper functions
        Row<int> initialize() {
            Row<int> S(p, fill::zeros);
            if (initialization == "Random") {
                for (int i = 0; i < s; ++i) {
                    int randIdx = rand() % p; // Random index between 0 and p-1
                    while (S(randIdx) == 1) { // sample without placement
                        randIdx = rand() % p; 
                    };
                    S(randIdx) = 1;
                };
            } else {
                cerr << "Invalid data type " << endl;
                exit(1);
            };
            initial_state = Row<int>(S); // copy S to initial_state

            return S;
        };

        Row<int> draw_q(Row<int> &S) {
            // s = sum(S);
            Row<int> S_new = S;
            uvec idx_0 = arma::find(S_new == 0);
            uvec idx_1 = arma::find(S_new == 1);
            S_new(idx_0(rand() % idx_0.n_elem)) = 1;
            S_new(idx_1(rand() % idx_1.n_elem)) = 0;

            return S_new;
        };

        Col<double> OLS_pred(Row<int> &S, string method = "fast") {
            uvec idx_1 = arma::find(S == 1);
            mat X_S = X.cols(idx_1);
            vec beta = arma::zeros(p); // init beta
            // fit linear regression and find predicted y
            if (method == "fast") {
                beta = solve(X_S.t() * X_S, X_S.t() * y); // solve_opts::fast
            }
            else if (method == "likely_sympd") {
                beta = solve(X_S.t() * X_S, X_S.t() * y, solve_opts::likely_sympd);
            } else if (method == "LU") { // LU decomposition
                mat L, U;
                lu(L, U, X_S.t() * X_S);
                beta = solve(U, solve(L, X_S.t() * y)); // U^-1(L^-1 * Xy)
            }
            vec Y_S = X_S * beta;

            return Y_S;
        };

        double pi_n(Col<double> &Y_S) {
            double Du = sensitivity_scaling;
            double log_pi_n = (-1) * epsilon * pow(arma::norm(y-Y_S), 2) / Du; // update Du from 2*Du

            return log_pi_n;
        };

        void MCMC() { // MCMC without checking S has already been seen
            S = initialize();
            Col<double> Y_S = OLS_pred(S);
            double log_pi_n = pi_n(Y_S);
            double norm_y = pow(arma::norm(y), 2);
            RSS = pow(arma::norm(y-Y_S), 2) / norm_y;

            log_pi_n_list.push_back(log_pi_n);
            RSS_list.push_back(RSS);

            int no_acceptances = 0;
            double time_cost = 0.0;

            for(int i = 0; i < max_iter; ++i) {
                // proposal draw
                Row<int> S_new = draw_q(S);
                double log_pi_n_new = 0; // def as 0 or log_pi_n_new, init

                vec Y_S_new = OLS_pred(S_new, "fast");
                log_pi_n_new = pi_n(Y_S_new);
                log_pi_n_list.push_back(log_pi_n_new);

                // compute hastings ratio
                double HR = exp(log_pi_n_new - log_pi_n);
                double R = std::min(1.0, HR);
                double unif_value = rand() / double(RAND_MAX); // unif[0,1]
                if (unif_value <= R) {
                    // accept
                    S = S_new; // update current S
                    log_pi_n = log_pi_n_new;
                    no_acceptances += 1;
                    // update RSS value
                    RSS = pow(norm(y-Y_S_new), 2) / norm_y;
                };
                RSS_list.push_back(RSS); // push the RSS

                // record the last # S, # = F1_score_size
                if(i >= (max_iter - F1_score_size)) {
                    S_list.push_back(S_new);
                };
            };
            acceptance = no_acceptances;
        };

};

double F1_score(vector<int> vec_S, vector<int> vec_S_hat) {
    vector<int> S_intersect;
    set_intersection(vec_S_hat.begin(), vec_S_hat.end(), vec_S.begin(), 
                        vec_S.end(), back_inserter(S_intersect));
    double prec = double(S_intersect.size()) / double(max(1, int(vec_S_hat.size())));
    double recall = double(S_intersect.size()) / double(vec_S.size());
    double F1 = 0.0;
    if (prec + recall > 0) F1 = 2*(prec*recall) / (prec+recall);

    return F1;
};

mat randmvn(vec mu, mat Sigma, int n) {
    int p = Sigma.n_cols;
    mat Y = randn(n, p);
    mat out = arma::repmat(mu, 1, n).t() + Y * arma::chol(Sigma);
    
    return out;
};

vector<double> fit_MCMC(DP_BSS model, mat X, vec y, vec beta, int i) {
    // switch different seeds to get different results for each chain
    srand(42+i); // srand(42+i*5)

    // fit model
    model.fit(X, y);
    vector<double> Everything = model.RSS_list; // Everything contains RSS and F1 scores

    // convert to array
    uvec S = arma::find(beta != 0); 
    vector<int> vec_S = conv_to<vector<int>>::from(S);

    // calculate F1 scores
    for (Row<int> S : model.S_list) {
        uvec S_hat = arma::find(S > 0);
        vector<int> vec_S_hat = conv_to<vector<int>>::from(S_hat);
        double F1 = F1_score(vec_S, vec_S_hat);
        Everything.push_back(F1);
    };

    return Everything;
};


int main() { // parallel
    // set seed
    srand(42);
    arma::arma_rng::set_seed(42);

    // generate random data
    int n = 900;
    int p = 2000;
    int s = 10;
    double rho = 0.3;
    string data_type = "AR1"; // Uniform or AR1
    string signal  = "weak"; // "strong" or "weak"

    int eps = 3;
    int sparse = 10;
    double sens_scale = 20; // updated later
    int max_iteration = 1000000; // 10^6
    int F1_score_size = 10; // size of F1 score list
    bool standardization = true;
    string initial = "Random"; // only "Random" supported
    int chain_id = atoi(getenv("SLURM_ARRAY_TASK_ID")); // char* to int

    // X and error
    mat X; // initialize
    vec e;
    // mat Sigma = (1-rho) * arma::eye(p, p) + rho * arma::ones(p, p);
    if (data_type == "Uniform") {
        X = 2.0 * arma::randu(n, p) - 1.0; // cast to unnif[-1, 1]
        e = 2.0 * randu(n) - 1.0; // unif[-1, 1]
        // e = 0.2 * randu(n) - 1.0; // unif[-0.1, 0.1]
    } else if (data_type == "AR1") {
        vec vec_rho(p, fill::value(rho));
        vec first_col = arma::pow(vec_rho, arma::linspace(0, p-1, p));
        mat Sigma = toeplitz(first_col);
        vec mu = zeros(p); // zero col_vec wiht length p
        X = randmvn(mu, Sigma, n);
        // normalize for each feature/column
        rowvec colmean = mean(X, 0); // dim = 0
        rowvec colstddev = stddev(X, 0, 0); // norm_type = 0, dim = 0
        for (uword i=1; i<X.n_cols; ++i) {
            X.col(i) -= colmean(i);
            X.col(i) /= colstddev(i);
        };
        e = 0.2 * randu(n) - 1.0; // unif[-0.1, 0.1]
    } else {
        cerr << "Invalid data type " << endl;
        return -1;
    }
    
    // beta
    vec beta = arma::zeros(p);
    if (signal == "weak") {  // we set first s elements as non-zeros
        beta.head(s).fill(2*pow(1*log(p)/n, 0.5));
        // beta.head(s).fill(0.5*pow(1*log(p)/n, 0.5));
    } else { // signal "strong"
        beta.head(s).fill(2*pow(s*log(p)/n, 0.5)); 
        // beta.head(s).fill(pow(0.5*s*log(p)/n, 0.5)); 
    };
    // y
    vec y = X * beta + e;

    // export X, y, beta
    char cwd[FILENAME_MAX];
    getcwd(cwd, sizeof(cwd));
    string current_path(cwd);
    string root = current_path + "/" + data_type;
    int root_status = mkdir(root.c_str(), 0777);
    cout << "Root Status: " << root_status << "; Root: " << root << endl; 
    string folder = "/eps_" + to_string(eps) + "_iter_" + to_string(max_iteration) + "_" + signal;
    string dir = root + folder;
    int dir_status = mkdir(dir.c_str(), 0777);;// make folder
    cout << "Dir Status: " << dir_status << "; Dir: " << dir << endl; 

    string X_path = dir + "/data_X.csv";
    string y_path = dir + "/data_y.csv";
    string beta_path = dir + "/data_beta.csv";
    X.save(X_path, csv_ascii);
    y.save(y_path, csv_ascii);
    beta.save(beta_path, csv_ascii);

    // calculate sensensitivty scale
    int J = 100;
    double v = 0.0;
    for (int i = 0; i < 4000; ++i) {
        Row<int> S(p, fill::zeros);
        int randIdx = rand() % p; // Random index between 0 and p-1
        while (S(randIdx) == 1) { // sample without placement
            randIdx = rand() % p; 
        };
        S(randIdx) = 1;
        uvec idx_1 = arma::find(S == 1);
        mat X_ = X.cols(idx_1);

        mat X_temp(J, p, fill::randu);
        mat X_new = 2.0 * X_temp - 1.0;
        vec e = 2.0 * randu(J) - 1.0;
        vec y_new = X_new * beta + e;
        vec beta_hat = solve((X_.t() * X_)/n, (X_.t() * y)/n);
        vec y_new_hat = X_new.cols(idx_1) * beta_hat;
        double v_temp = max(pow(y_new - y_new_hat, 2));
        if (v_temp > v) v = v_temp; // max v;
    }
    double Du_hat_dbl = v;
    sens_scale = round(Du_hat_dbl * 100.0) / 100.0; // round to 2 decimal point


    // print job start
    cout << "Chain ID: " << chain_id <<  "; Number of Iteration: " << max_iteration << endl;
    cout << "Sensitivity Scale: " << sens_scale << endl;
    
    // initialize model
    DP_BSS model;
    model.init(sparse, eps, sens_scale, max_iteration, standardization, initial, F1_score_size);

    clock_t start = clock();
    vector<double> result = fit_MCMC(model, X, y, beta, chain_id);
    result.push_back(sens_scale); // structure: RSS, F1, sens_scale;
    clock_t end = clock();
    double time_cost = static_cast<double>(end - start) / (double)CLOCKS_PER_SEC;
    cout << "Fitting time cost: " << time_cost << "s\n";

    
    string RSS_filename = "/chain_" + to_string(chain_id) + "_eps_" + to_string(eps) + "_iter_" + to_string(max_iteration) + ".csv";
    string RSS_path = dir + RSS_filename;
    string F1_filename = "/F1_chain_" + to_string(chain_id) + "_eps_" + to_string(eps) + "_iter_" + to_string(max_iteration) + ".csv";
    string F1_path = dir + F1_filename;

    std::ofstream out1(RSS_path);
    if(out1.is_open()) {
        for (size_t i = 0; i < result.size()-F1_score_size-1; ++i) {
            out1 << result[i] << endl;
        };
        out1.close();
    };

    std::ofstream out2(F1_path);
    if(out2.is_open()) {
        for (size_t i = result.size()-F1_score_size-1; i < result.size(); ++i) {
            out2 << result[i] << endl;
        };
        out2.close();
    };


    return 0;
}