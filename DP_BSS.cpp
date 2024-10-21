// DP-BSS cpp

#define  ARMA_USE_LAPACK
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <armadillo>
#include <random>
#include <vector>
#include <deque>
#include <numeric>
#include <ctime>
#include <gurobi_c++.h>

// For path and directory
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace arma;

class ebreg {
    public:
        // Initialization
        void init(int sparse, double eps, double sens_scale, int max_iteration,
                    bool standardization, string init, int F1_size, double K) {
            s = sparse;
            epsilon = eps;
            sensitivity_scaling = sens_scale;
            max_iter = max_iteration;
            standardize = standardization;
            initialization = init;
            F1_score_size = F1_size;
            K = K;
        };

        // other parameter
        Row<int> S;
        // vector<Row<int>> S_list; // store last # F1_score_size S
        deque<Row<int>> S_list;
        vector<double> log_pi_n_list;
        vector<double> RSS_list;

        // Functions
        void fit(arma::Mat<double>& X_in, arma::Col<double>& y_in, std::mt19937& gen) {
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
            MCMC(gen);
        };

    
    private:
        // tuning parameters
        int s = 4;
        double epsilon = 3.0;
        double sensitivity_scaling = 20.0;

        // MCMC parameters
        int max_iter = 1e6;

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
        double RSS;
        int F1_score_size;
        double K = 0.5; // l1-ball radius
        double l1_opt_eps = 1e-8; // stop crieria for l1_opt_pred
        double lr = 0.001; // initial learning rate for l1_opt_pred
        double l1_opt_max_iter = 100; 

        // helper functions
        Row<int> initialize(mt19937& gen) {
            Row<int> S(p, fill::zeros);
            if (initialization == "Random") {
                uniform_int_distribution<> dist_p(0, p - 1);
                for (int i = 0; i < s; ++i) {
                    int randIdx = dist_p(gen);
                    while (S(randIdx) == 1) { // sample without placement
                        randIdx = dist_p(gen);
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

        Row<int> draw_q(Row<int> &S, mt19937& gen) {
            
            Row<int> S_new = S;
            uvec idx_0 = arma::find(S_new == 0);
            uvec idx_1 = arma::find(S_new == 1);
            uniform_int_distribution<> dist_zero(0, idx_0.n_elem - 1);
            uniform_int_distribution<> dist_one(0, idx_1.n_elem - 1);
            int zero_index = idx_0(dist_zero(gen));
            int one_index = idx_1(dist_one(gen));
            S_new(one_index) = 0;
            S_new(zero_index) = 1;


            // cout << "Print S_new: " << arma::find(S_new == 1) << endl;
            return S_new;
        };

        double pi_n(Col<double> &Y_S) {
            double Du = sensitivity_scaling;
            double log_pi_n = (-1) * epsilon * pow(arma::norm(y-Y_S), 2) / Du; // update Du from 2*Du

            return log_pi_n;
        };

        
        Col<double> solve_beta(mat X, vec y, double z) {
            int m = X.n_rows; // Number of data points
            int n = X.n_cols; // Number of variables

            // Create an environment
            GRBEnv env = GRBEnv(true);
            // env.set("LogFile", "gurobi_opt.log");
            env.set("OutputFlag", "0");
            env.start();

            // Create an empty model
            GRBModel model = GRBModel(env);
            model.set(GRB_IntParam_Threads, 4); // multi-cores
            model.set(GRB_DoubleParam_OptimalityTol, 1e-6); // less strict convergence

            // Create Gurobi variables
            GRBVar beta[n];
            for (int i = 0; i < n; ++i) {
                // beta[i] = model.addVar(-z, z, 0.0, GRB_CONTINUOUS, "beta_" + std::to_string(i));
                beta[i] = model.addVar(0.0, z, 0.0, GRB_CONTINUOUS, "beta_" + std::to_string(i));
            }


            // Set objective function
            GRBQuadExpr obj = 0.0;
            for (int i = 0; i < m; ++i) {
                GRBLinExpr row = 0.0;
                for (int j = 0; j < n; ++j) {
                    row += X(i, j) * beta[j];
                }
                obj += (row - y(i)) * (row - y(i));
            }
            model.setObjective(obj, GRB_MINIMIZE);

            GRBLinExpr l1_norm = 0.0;
            for (int i = 0; i < n; ++i) {
                // l1_norm += abs_beta[i];
                l1_norm += beta[i];
            }
            model.addConstr(l1_norm <= z, "l1_norm");

            // optimize
            // std::cout << "start optimizing ..." << std::endl;
            model.optimize();

            vec beta_opt(n);
            for (int i = 0; i < n; ++i) {
                beta_opt(i) = beta[i].get(GRB_DoubleAttr_X);
            }
            return beta_opt;
        };

        Col<double> L1_Opt_Pred_GRB(Row<int> &S, double K) {
            uvec idx_1 = arma::find(S == 1);
            mat X_S = X.cols(idx_1);
            vec beta = solve_beta(X_S, y, K);
            vec Y_S = X_S * beta;
            // beta.print();
            return Y_S;
        };

        void MCMC(mt19937& gen) { // MCMC without checking S has already been seen
            S = initialize(gen);
            Col<double> Y_S = L1_Opt_Pred_GRB(S, K);
            double log_pi_n = pi_n(Y_S);
            double norm_y = pow(arma::norm(y), 2);
            RSS = std::min(1.0, pow(arma::norm(y-Y_S), 2) / norm_y);

            log_pi_n_list.push_back(log_pi_n);
            RSS_list.push_back(RSS);
            double time_cost = 0.0;

            for(int i = 0; i < max_iter; ++i) {
                // proposal draw
                Row<int> S_new = draw_q(S, gen);
                double log_pi_n_new = 0; // def as 0 or log_pi_n_new, init

                vec Y_S_new = L1_Opt_Pred_GRB(S_new, K);
                log_pi_n_new = pi_n(Y_S_new);
                log_pi_n_list.push_back(log_pi_n_new);

                // compute hastings ratio
                double HR = exp(log_pi_n_new - log_pi_n);
                double R = std::min(1.0, HR);
                uniform_real_distribution<> unif_0_1(0.0, 1.0);
                double unif_value = unif_0_1(gen);
                if (unif_value <= R) {
                    // accept
                    S = S_new; // update current S
                    log_pi_n = log_pi_n_new;
                    // update RSS value
                    RSS = std::min(1.0, pow(norm(y-Y_S_new), 2) / norm_y);

                    // record the last # S, # = F1_score_size
                    if (S_list.size() >= F1_score_size) {
                        S_list.pop_front();
                        S_list.push_back(S_new);
                    } else {
                        S_list.push_back(S_new);
                    };
                };
                RSS_list.push_back(RSS); // push the RSS
            };
        };

};

double F1_score(vector<int> vec_S, vector<int> vec_S_hat) {
    vector<int> S_intersect;
    set_intersection(vec_S_hat.begin(), vec_S_hat.end(), vec_S.begin(), 
                        vec_S.end(), back_inserter(S_intersect));
    // print vec_S_hat
    cout << "vec_S_hat:";
    for (int num : vec_S_hat) {
            cout << num << " ";
    }
    cout << endl;
    
    // print vec_S
    cout << "vec_S:";
    for (int num : vec_S) {
            cout << num << " ";
    }
    cout << endl;

    // print S_intersect
    cout << "S_intersect:";
    for (int num : S_intersect) {
            cout << num << " ";
    }
    cout << endl;

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

vector<double> fit_MCMC(ebreg model, mat X, vec y, vec beta, int i) {
    // switch different seeds to get different results for each chain
    srand(42+i); // srand(42+i*5)
    std::mt19937 gen(42+i);
    // fit model
    model.fit(X, y, gen);
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

std::string double_to_string(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

int main(int argc, char* argv[]) { // parallel
    // set seed
    srand(42);
    arma::arma_rng::set_seed(42);

    // generate random data
    int n = 900;
    int p = 2000;
    int sparse = 4;
    double rho = 0.3;
    string data_type = "AR1"; // "Uniform" or "AR1"
    string signal  = "strong"; // "strong" or "weak"

    double eps = 0.5;
    double K = 3.5;
    double sens_scale = 20; // updated later
    int max_iteration = 1e4; // 1e5 // 10^6
    int F1_score_size = 1; // size of F1 score list
    bool standardization = true;
    string initial = "Random"; // only "Random" supported
    // int chain_id = atoi(getenv("SLURM_ARRAY_TASK_ID")); // char* to int
    int chain_id = atoi(argv[1]);


    // dir
    char cwd[FILENAME_MAX];
    getcwd(cwd, sizeof(cwd));
    string current_path(cwd);
    string root = current_path + "/" + data_type;
    int root_status = mkdir(root.c_str(), 0777);
    cout << "Root Status: " << root_status << "; Root: " << root << endl;
    // folder "eps_3_iter_1000000_K_0.50"
    string folder = "/eps_" + double_to_string(eps, 1) + "_iter_" + to_string(max_iteration) +
        "_" + signal + "_K_" + double_to_string(K, 2) + "_s_" + to_string(sparse);
    string dir = root + folder;
    int dir_status = mkdir(dir.c_str(), 0777);// make folder
    cout << "Dir Status: " << dir_status << "; Dir: " << dir << endl;
    string X_path = dir + "/data_X.csv";
    string y_path = dir + "/data_y.csv";
    string beta_path = dir + "/data_beta.csv";
    mat X;
    vec y;
    vec beta;
    if (dir_status == -1) { // dir exist, load X, y
        X.load(X_path, csv_ascii);
        y.load(y_path, csv_ascii);
        beta.load(beta_path, csv_ascii);
    } else { // generate data
        vec e;
        // mat Sigma = (1-rho) * arma::eye(p, p) + rho * arma::ones(p, p);
        if (data_type == "Uniform") {
            X = 2.0 * arma::randu(n, p) - 1.0; // cast to unnif[-1, 1]
            e = 0.2 * randu(n) - 0.1; // unif[-0.1, 0.1]
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
            e = 0.2 * randu(n) - 0.1; // unif[-0.1, 0.1]
        } else {
            cerr << "Invalid data type " << endl;
            return -1;
        };
        
        // beta
        beta = arma::zeros(p);
        if (signal == "weak") {  // we set first s elements as non-zeros
            beta.head(sparse).fill(2*pow(log(p)/n, 0.5));
            // beta.head(s).fill(0.5*pow(1*log(p)/n, 0.5));
        } else { // signal "strong"
            beta.head(sparse).fill(2*pow(sparse*log(p)/n, 0.5)); 
            // beta.head(s).fill(pow(0.5*s*log(p)/n, 0.5)); 
        };
        // y
        y = X * beta + e;

        // export X, y, beta
        X.save(X_path, csv_ascii);
        y.save(y_path, csv_ascii);
        beta.save(beta_path, csv_ascii);
    };
    
    sens_scale = pow((K*1.0 + max(abs(y))), 2);


    // print job start
    cout << "Chain ID: " << chain_id <<  "; Number of Iteration: " << max_iteration << endl;
    cout << "Sensitivity Scale: " << sens_scale << endl;
    
    // initialize model
    ebreg model;
    model.init(sparse, eps, sens_scale, max_iteration, standardization, initial, F1_score_size, K);

    clock_t start = clock();
    vector<double> result = fit_MCMC(model, X, y, beta, chain_id);
    result.push_back(sens_scale); // structure: RSS, F1, sens_scale;
    clock_t end = clock();
    double time_cost = static_cast<double>(end - start) / (double)CLOCKS_PER_SEC;
    cout << "Fitting time cost: " << time_cost << "s\n";

    
    string RSS_filename = "/chain_" + to_string(chain_id) + "_eps_" + double_to_string(eps,1) + "_iter_" + to_string(max_iteration) + ".csv";
    string RSS_path = dir + RSS_filename;
    string F1_filename = "/F1_chain_" + to_string(chain_id) + "_eps_" + double_to_string(eps,1) + "_iter_" + to_string(max_iteration) + ".csv";
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