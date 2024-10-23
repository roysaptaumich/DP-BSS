# DP-BSS
Metropolis-Hastings algorithm for differentially private best subset selection algorithms proposed in this [paper](https://arxiv.org/abs/2310.07852).

## Usage for python version

**Requirements**
```
numpy==1.26.3
scipy==1.11.4
matplotlib==3.8.0
scikit-learn==1.3.0
abess==0.4.6
joblib==1.2.0
```

**Run Examples**
```
python main.py
```

## Usage for cplusplus version

**Requirements**
```
gcc/10.3.0
armadillo/11.4.2
lapack/3.10.1
gurobi/10.0.2
```

**Run Examples**
```
g++ DP_BSS.cpp -o DP_BSS_Expr -O2 \
    -I $ARMA_INC -DARMA_DONT_USE_WRAPPER \
    -I$GUROBI_HOME/include -L$GUROBI_HOME/lib \
    -L $OPENBLAS_ROOT \
    -lopenblas -lgurobi_g++5.2 -lgurobi100

./DP_BSS_Expr 1 # chain_number

python load_cpp_result 0.5 10000 strong 1 Uniform 3.5 4 # eps, iter, signal, num_chain, data_type, K, sparse
```