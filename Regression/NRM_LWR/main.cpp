//
//  main.cpp
//  NRM_LWR - Newton-Raphson Method with Locally Weighted Regression
//
//  Created by Zhalgas Baibatyr on 1/20/16.
//  Copyright © 2016 Zhalgas Baibatyr. All rights reserved.
//

/*

    DEGREE ----- the degree of a polynomial in regression function
    N_SAMPLES -- number of training samples
    N_FEATURES - number of given input features
    BANDWIDTH -- controls how quickly the weight of a training sample
                 falls off with distance of its features from the query point
    Hessian ---- Hessian matrix
    features --- input features, including intercept term (xₒ = 1)
    target ----- target (output) values
    params ----- parameters (weights) with initial values (zeros)
    hypothesis - hypothesis function
    dCost ------ derivative of a cost function (LMS)
    weights ---- non-negative valued weights for LMS
    qPoint ----- particular query point

*/

#include <armadillo>

using namespace arma;

int main()
{
    /* Loading initial data: */
    mat initData;
    initData.load("../data/regression1", arma_ascii);

    /* Initializing constants: */
    const uword DEGREE = 5;
    const uword N_SAMPLES  = initData.n_rows;
    const uword N_FEATURES = initData.n_cols - 1;
    const double BANDWIDTH = 0.2;

    /* Transforming input data using polynomial formula: */
    mat features(N_SAMPLES, DEGREE * N_FEATURES + 1);
    features.col(0) = ones<vec>(N_SAMPLES);
    for (int i = 0; i < N_FEATURES; ++i)
    {
        for (int j = 1; j <= DEGREE; ++j)
        {
            features.col(i * DEGREE + j) = pow(initData.col(i), j);
        }
    }

    /* Preparing other data: */
    vec target = initData.col(initData.n_cols - 1);
    vec params  = zeros<vec>(features.n_cols);
    vec hypothesis;
    vec dCost;
    mat Hessian(features.n_cols, features.n_cols);
    vec weights;

    /* Random query point generation: */
    arma_rng::set_seed_random();
    rowvec qPoint(features.n_cols);
    qPoint(0) = 1;
    vec randNums = randu<vec>(N_FEATURES);
    std::cout << "Random query point: " << randNums << std::endl;
    for (int i = 0; i < N_FEATURES; ++i)
    {
        for (int j = 1; j <= DEGREE; ++j)
        {
            qPoint(i * DEGREE + j) = pow(randNums(i), j);
        }
    }

    wall_clock timer;
    double elapsedTime;
    timer.tic();

    /* NRM_LWR performs here: */
    weights = exp(-(sum(pow(features - repmat(qPoint, N_SAMPLES, 1), 2), 1) / features.n_cols) / (2 * pow(BANDWIDTH, 2)));
    Hessian = features.t() * features;
    while (true)
    {
        /* Calculating hypothesis values and updating parameters: */
        hypothesis = features * params;
        dCost = features.t() * (weights % (hypothesis - target)) / N_SAMPLES;
        params -= Hessian.i() * dCost;
     /* params -= solve(Hessian, dCost); */

        /* Checking how close to the minimum: */
        if (norm(dCost) < 1e-5)
        {
            hypothesis = features * params;
            break;
        }
    }

    /* Measuring the performance of the algorithm: */
    elapsedTime = timer.toc();
    printf("Elapsed time: %f sec.\n\n", elapsedTime);

    mat outputData = join_rows(initData.cols(0, initData.n_cols - 2), hypothesis);
    outputData.save("outputData", arma_ascii);
    hypothesis.save("hypothesis", arma_ascii);
    params.save("params", arma_ascii);

    return EXIT_SUCCESS;
}
