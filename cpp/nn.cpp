// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Demonstrate Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     Showcases how to efficiently evaluate privacy-perserving neural networks.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint paper
// ----------------------------------------------------------------------------|


// Includes
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/time.h>
// Multi-processing
#include <sys/wait.h>
#include <unistd.h>
// tfhe-lib
#include "tfhe.h"
#include "tfhe_garbage_collector.h"
#include "tlwe.h"
#include "tgsw.h"
#include "lwesamples.h"
#include "lwekey.h"
#include "lweparams.h"
#include "polynomials.h"


// Defines
#define VERBOSE 1
#define STATISTICS true
#define WRITELATEX true
#define N_PROC 4

// Security constants
#define SECLEVEL 80
#define SECNOISE true
#define SECALPHA pow(2., -20)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  600                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS 10           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params

// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_LAYERS 3
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10

#define CARD_TESTSET 10000

// Files are expected in the executable's directory
#define PATH_TO_FILES       "buildotests/test/" //TODO FIXME!
#define FILE_TXT_IMG        "txt_img_test.txt"
#define FILE_TXT_BIASES     "txt_biases.txt"
#define FILE_TXT_WEIGHTS    "txt_weights.txt"
#define FILE_TXT_LABELS     "txt_labels.txt"
#define FILE_LATEX          "results_LaTeX.tex"
#define FILE_STATISTICS     "results_stats.txt"

// Tweak neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100

#define MSG_SLOTS    700
#define TORUS_SLOTS  400


using namespace std;

void deleteTensor(int*** tensor, int dim_mat, const int* dim_vec);
void deleteMatrix(int**  matrix, int dim_mat);


/// Generate gate bootstrapping parameters for FHE_NN
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda)
{
    if (minimum_lambda > 128)
        cerr << "Sorry, for now, the parameters are only implemented for about 128bit of security!\n";

    static const int n = SEC_PARAMS_n;
    static const int N = SEC_PARAMS_N;
    static const int k = SEC_PARAMS_k;
    static const double max_stdev = SEC_PARAMS_STDDEV;

    static const int bk_Bgbit    = SEC_PARAMS_BK_BASEBITS;  //<-- ld, thus: 2^10
    static const int bk_l        = SEC_PARAMS_BK_LENGTH;
    static const double bk_stdev = SEC_PARAMS_BK_STDDEV; 

    static const int ks_basebit  = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1
    static const int ks_length   = SEC_PARAMS_KS_LENGTH;
    static const double ks_stdev = SEC_PARAMS_KS_STDDEV;


    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}


int main(int argc, char **argv)
{
    // Security
    const int minimum_lambda = SECLEVEL;
    const bool noisyLWE      = SECNOISE;
    const double alpha       = SECALPHA;

    // Input data
    const int n_images = CARD_TESTSET;

    // Network specific
    const int num_wire_layers = NUM_NEURONS_LAYERS - 1;
    const int num_neuron_layers = NUM_NEURONS_LAYERS;
    const int num_neurons_in = NUM_NEURONS_INPUT;
    const int num_neurons_hidden = NUM_NEURONS_HIDDEN;
    const int num_neurons_out = NUM_NEURONS_OUTPUT;

    // Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
    const int topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

    const int space_msg = MSG_SLOTS;
    const int space_after_bs = TORUS_SLOTS;

    const bool clamp_biases  = false;
    const bool clamp_weights = false;

    const bool statistics        = STATISTICS;
    const bool writeLaTeX_result = WRITELATEX;

    const int threshold_biases  = THRESHOLD_WEIGHTS;
    const int threshold_weights = THRESHOLD_WEIGHTS;
    const int threshold_scores  = THRESHOLD_SCORE;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);

    const int total_num_hidden_neurons = n_images * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images
    const double avg_bs  = 1./NUM_NEURONS_HIDDEN;
    const double avg_total_bs  = 1./total_num_hidden_neurons;
    const double avg_img = 1./n_images;
    const double clocks2seconds = 1. / CLOCKS_PER_SEC;
    const int slice = (n_images+N_PROC-1)/N_PROC;

    // Huge arrays
    int*** weights = new int**[num_wire_layers];  // allocate and fill matrices holding the weights
    int ** biases  = new int* [num_wire_layers];  // allocate and fill vectors holding the biases
    int ** images  = new int* [n_images];
    int  * labels  = new int  [n_images];

    // Temporary variables
    string line;
    int el, l;
    int num_neurons_current_layer_in, num_neurons_current_layer_out;


    if (VERBOSE)
    {
        cout << "Starting experiment to classify " << n_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl;
        cout << "(Run: " << argv[0] << " )" << endl;
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << N_PROC << endl;
    }

    if (VERBOSE) cout << "Generate parameters for a security level (according to  CGGI16a) of at least " << minimum_lambda << " [bit]." << endl;
    TFheGateBootstrappingParameterSet *params = our_default_gate_bootstrapping_parameters(minimum_lambda);
    //TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(minimum_lambda);
    const LweParams *in_out_params   = params->in_out_params;

    if (VERBOSE) cout << "Generate the secret keyset." << endl;
    TFheGateBootstrappingSecretKeySet *secret = new_random_gate_bootstrapping_secret_keyset(params);
    const LweBootstrappingKeyFFT *bs_key = secret->cloud.bkFFT;


    if (VERBOSE) cout << "IMPORT PIXELS, WEIGHTS, BIASES, and LABELS FROM FILES" << endl;
    if (VERBOSE) cout << "Reading images (regardless of dimension) from " << FILE_TXT_IMG << endl;
    ifstream file_images(FILE_TXT_IMG);

    for (int img=0; img<n_images; ++img)
        images[img] = new int[num_neurons_in];

    int filling_image = 0;
    int image_count = 0;
    while(getline(file_images, line))
    {
        images[filling_image][image_count++] = stoi(line);
        if (image_count == num_neurons_in)
        {
            image_count = 0;
            filling_image++;
        }
    }
    file_images.close();


    if (VERBOSE) cout << "Reading weights from " << FILE_TXT_WEIGHTS << endl;
    ifstream file_weights(FILE_TXT_WEIGHTS);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        weights[l] = new int*[num_neurons_current_layer_in];
        for (int i = 0; i<num_neurons_current_layer_in; ++i)
        {
            weights[l][i] = new int[num_neurons_current_layer_out];
            for (int j=0; j<num_neurons_current_layer_out; ++j)
            {
                getline(file_weights, line);
                el = stoi(line);
                if (clamp_weights)
                {
                    if (el < -threshold_weights)
                        el = -threshold_weights;
                    else if (el > threshold_weights)
                        el = threshold_weights;
                    // else, nothing as it holds that: -threshold_weights < el < threshold_weights
                }
                weights[l][i][j] = el;
            }
        }
    }
    file_weights.close();


    if (VERBOSE) cout << "Reading biases from " << FILE_TXT_BIASES << endl;
    ifstream file_biases(FILE_TXT_BIASES);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        biases [l] = new int [num_neurons_current_layer_out];
        for (int j=0; j<num_neurons_current_layer_out; ++j)
        {
            getline(file_biases, line);
            el = stoi(line);
            if (clamp_biases)
            {
                if (el < -threshold_biases)
                    el = -threshold_biases;
                else if (el > threshold_biases)
                    el = threshold_biases;
                // else, nothing as it holds that: -threshold_biases < el < threshold_biases
            }
            biases[l][j] = el;
        }
    }
    file_biases.close();


    if (VERBOSE) cout << "Reading labels from " << FILE_TXT_LABELS << endl;
    ifstream file_labels(FILE_TXT_LABELS);
    for (int img=0; img<n_images; ++img)
    {
        getline(file_labels, line);
        labels[img] = stoi(line);
    }
    file_labels.close();

    if (VERBOSE) cout << "Import done. END OF IMPORT" << endl;



    // Temporary variables and Pointers to existing arrays for convenience
    bool notSameSign;
    Torus32 mu, phase;

    int** weight_layer;
    int * bias;
    int * image;
    int pixel, label;
    int x, w, w0;

    LweSample *multi_sum, *enc_image, *bootstrapped;


    int multi_sum_clear[num_neurons_hidden];
    int output_clear   [num_neurons_out];

    int max_score = 0;
    int max_score_clear = 0;
    int class_enc = 0;
    int class_clear = 0;
    int score = 0;
    int score_clear = 0;


    bool failed_bs = false;
    // Counters
    int count_errors = 0;
    int count_errors_with_failed_bs = 0;
    int count_disagreements = 0;
    int count_disagreements_with_failed_bs = 0;
    int count_disag_pro_clear = 0;
    int count_disag_pro_hom = 0;
    int count_wrong_bs = 0;

    int r_count_errors, r_count_disagreements, r_count_disag_pro_clear, r_count_disag_pro_hom, r_count_wrong_bs, r_count_errors_with_failed_bs, r_count_disagreements_with_failed_bs;
    double r_total_time_network, r_total_time_bootstrappings;

    // For statistics output
    double avg_time_per_classification = 0.0;
    double avg_time_per_bootstrapping = 0.0;
    double total_time_bootstrappings = 0.0;
    double total_time_network = 0.0;
    double error_rel_percent = 0.0;

    // Timings
    clock_t bs_begin, bs_end, net_begin, net_end;
    double time_per_classification, time_per_bootstrapping, time_bootstrappings;

    pid_t pids[N_PROC];
    int pipes[N_PROC][2];

    for (int id_proc=0; id_proc < N_PROC; ++id_proc)
    {
        pipe(pipes[id_proc]);   // before fork!
        pid_t pid = fork();
        if (pid != 0)
        {
            pids [id_proc] = pid;
            close(pipes[id_proc][1]);
        }
        else
        {
            close(pipes[id_proc][0]);
            for (int img = id_proc*slice; img < ( (id_proc+1)*slice) && (img< n_images); /*img*/ )
            {
                image = images[img];
                label = labels[img];
                ++img;

                // Generate encrypted inputs for NN (LWE samples for each image's pixels on the fly)
                // To be generic...
                num_neurons_current_layer_out= topology[0];
                num_neurons_current_layer_in = num_neurons_current_layer_out;

                enc_image = new_LweSample_array(num_neurons_current_layer_in, in_out_params);
                for (int i = 0; i < num_neurons_current_layer_in; ++i)
                {
                    pixel = image[i];
                    mu = modSwitchToTorus32(pixel, space_msg);
                    if (noisyLWE)
                    {
                        lweSymEncrypt(enc_image + i, mu, alpha, secret->lwe_key);
                    }
                    else
                    {
                        lweNoiselessTrivial(enc_image + i, mu, in_out_params);
                    }
                }

                // ========  FIRST LAYER(S)  ========
                net_begin = clock();

                multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params);
                for (l=0; l<num_wire_layers - 1 ; ++l)     // Note: num_wire_layers - 1 iterations; last one is special. Access weights from level l to l+1.
                {
                    // To be generic...
                    num_neurons_current_layer_in = num_neurons_current_layer_out;
                    num_neurons_current_layer_out= topology[l+1];
                    bias = biases[l];
                    weight_layer = weights[l];
                    for (int j=0; j<num_neurons_current_layer_out; ++j)
                    {
                        w0 = bias[j];
                        multi_sum_clear[j] = w0;
                        mu = modSwitchToTorus32(w0, space_msg);
                        lweNoiselessTrivial(multi_sum + j, mu, in_out_params);  // bias in the clear

                        for (int i=0; i<num_neurons_current_layer_in; ++i)
                        {
                            x = image [i];  // clear input
                            w = weight_layer[i][j];  // w^dagger
                            multi_sum_clear[j] += x * w; // process clear input
                            lweAddMulTo(multi_sum + j, w, enc_image + i, in_out_params); // process encrypted input
                        }
                    }
                }

                // Bootstrap multi_sum
                bootstrapped = new_LweSample_array(num_neurons_current_layer_out, in_out_params);
                bs_begin = clock();
                for (int j=0; j<num_neurons_current_layer_out; ++j)
                {
                    /**
                     *  Bootstrapping results in each coordinate 'bootstrapped[j]' to contain an LweSample
                     *  of low-noise (= fresh LweEncryption) of 'mu_boot*phase(multi_sum[j])' (= per output neuron).
                     */
                    tfhe_bootstrap_FFT(bootstrapped + j, bs_key, mu_boot, multi_sum + j);
                }
                bs_end = clock();
                time_bootstrappings = bs_end - bs_begin;
                total_time_bootstrappings += time_bootstrappings;
                time_per_bootstrapping = time_bootstrappings*avg_bs;
                if (VERBOSE) cout <<  time_per_bootstrapping*clocks2seconds << " [sec/bootstrapping]" << endl;

                delete_LweSample_array(num_neurons_current_layer_out, multi_sum);  // TODO delete or overwrite after use?

                failed_bs = false;
                for (int j=0; j<num_neurons_current_layer_out; ++j)
                {
                    phase = lwePhase(bootstrapped + j, secret->lwe_key);
                    notSameSign = multi_sum_clear[j]*t32tod(phase) < 0; // TODO adapt for non-binary case
                    if (notSameSign)
                    {
                        count_wrong_bs++;
                        failed_bs = true;
                    }
                }

                // ========  LAST (SECOND) LAYER  ========
                max_score = threshold_scores;
                max_score_clear = threshold_scores;

                bias = biases[l];
                weight_layer = weights[l];
                l++;
                num_neurons_current_layer_in = num_neurons_current_layer_out;
                num_neurons_current_layer_out= topology[l]; // l == L = 2
                multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params); // TODO possibly overwrite storage
                for (int j=0; j<num_neurons_current_layer_out; ++j)
                {
                    w0 = bias[j];
                    output_clear[j] = w0;
                    mu = modSwitchToTorus32(w0, space_after_bs);

                    lweNoiselessTrivial(multi_sum + j, mu, in_out_params);

                    for (int i=0; i<num_neurons_current_layer_in; ++i)
                    {
                        w = weight_layer[i][j];
                        lweAddMulTo(multi_sum + j, w, bootstrapped + i, in_out_params); // process encrypted input
                        // process clear input
                        if (multi_sum_clear[i] < 0)
                            output_clear[j] -= w;
                        else
                            output_clear[j] += w;

                    }
                    score = lwePhase(multi_sum + j, secret->lwe_key);
                    if (score > max_score)
                    {
                        max_score = score;
                        class_enc = j;
                    }
                    score_clear = output_clear[j];
                    if (score_clear > max_score_clear)
                    {
                        max_score_clear = score_clear;
                        class_clear = j;
                    }
                }

                if (class_enc != label)
                {
                    count_errors++;
                    if (failed_bs)
                        count_errors_with_failed_bs++;
                }

                if (class_clear != class_enc)
                {
                    count_disagreements++;
                    if (failed_bs)
                        count_disagreements_with_failed_bs++;

                    if (class_clear == label)
                        count_disag_pro_clear++;
                    else if (class_enc == label)
                        count_disag_pro_hom++;
                }
                net_end = clock();
                time_per_classification = net_end - net_begin;
                total_time_network += time_per_classification;
                if (VERBOSE) cout << "            "<< time_per_classification*clocks2seconds <<" [sec/classification]" << endl;
                // free memory
                delete_LweSample_array(num_neurons_in,     enc_image);
                delete_LweSample_array(num_neurons_hidden, bootstrapped);
                delete_LweSample_array(num_neurons_out,    multi_sum);

            }
            FILE* stream = fdopen(pipes[id_proc][1], "w");
            fprintf(stream, "%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", count_errors, count_disagreements, count_disag_pro_clear, count_disag_pro_hom, count_wrong_bs,
                    count_errors_with_failed_bs, count_disagreements_with_failed_bs, total_time_network, total_time_bootstrappings);
            fclose(stream);
            exit(0);
        }
    }

    for (auto pid : pids)
    {
        waitpid(pid, 0, 0);
    }

    time_per_classification = 0.0;
    time_per_bootstrapping = 0.0;
    for (int id_proc=0; id_proc<N_PROC; ++id_proc)
    {
        FILE* stream = fdopen(pipes[id_proc][0], "r");
        fscanf(stream, "%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", &r_count_errors, &r_count_disagreements,
               &r_count_disag_pro_clear, &r_count_disag_pro_hom, &r_count_wrong_bs, &r_count_errors_with_failed_bs,
               &r_count_disagreements_with_failed_bs, &r_total_time_network, &r_total_time_bootstrappings);
        fclose(stream);
        count_errors += r_count_errors;
        count_disagreements += r_count_disagreements;
        count_disag_pro_clear += r_count_disag_pro_clear;
        count_disag_pro_hom += r_count_disag_pro_hom;
        count_wrong_bs += r_count_wrong_bs;
        count_errors_with_failed_bs += r_count_errors_with_failed_bs;
        count_disagreements_with_failed_bs += r_count_disagreements_with_failed_bs;
        time_per_classification += r_total_time_network;
        time_per_bootstrapping += r_total_time_bootstrappings;
    }

    if (statistics)
    {
        ofstream of(FILE_STATISTICS);
        // Print some statistics
        error_rel_percent = count_errors*avg_img*100;
        avg_time_per_classification = time_per_classification*avg_img*clocks2seconds;
        avg_time_per_bootstrapping  = time_per_bootstrapping *avg_total_bs *clocks2seconds;

        cout << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        cout << "Disagreements: " << count_disagreements;
        cout << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        cout << "Wrong bootstrappings: " << count_wrong_bs << endl;
        cout << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        cout << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        cout << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        cout << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        of << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        of << "Disagreements: " << count_disagreements;
        of << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        of << "Wrong bootstrappings: " << count_wrong_bs << endl;
        of << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        of << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        of << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        of << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        // Write some statistics
        cout << "\n Wrote statistics to file: " << FILE_STATISTICS << endl << endl;
        of.close();
    }

    if (writeLaTeX_result)
    {
        cout << "\n Wrote LaTeX_result to file: " << FILE_LATEX << endl << endl;
        ofstream of(FILE_LATEX);
        of << "%\\input{"<<FILE_LATEX<<"}" << endl;

        of << "% Experiments detailed" << endl;
        of << "\\newcommand{\\EXPnumBS}{$"<<total_num_hidden_neurons<<"$}" << endl;
        of << "\\newcommand{\\EXPbsEXACT}{$"    <<avg_time_per_bootstrapping<<"$\\ [sec/bootstrapping]}" << endl;
        of << "\\newcommand{\\EXPtimeEXACT}{$"  <<avg_time_per_classification<<"$\\ [sec/classification]}" << endl;

        of << "\\newcommand{\\EXPnumERRabs}{$"  <<count_errors<<"$}" << endl;
        of << "\\newcommand{\\EXPnumERRper}{$"  <<error_rel_percent<<"\\ \\%$}" << endl;
        of << "\\newcommand{\\EXPwrongBSabs}{$" <<count_wrong_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPwrongDISabs}{$"<<count_disagreements_with_failed_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPdis}{$"        <<count_disagreements<<"$}" << endl;
        of << "\\newcommand{\\EXPclear}{$"      <<count_disag_pro_clear<<"$}" << endl;
        of << "\\newcommand{\\EXPhom}{$"        <<count_disag_pro_hom<<"$}" << endl << endl;

        of << "\\begin{Verbatim}[frame=single,numbers=left,commandchars=+\\[\\]%" << endl;
        of << "]" << endl;
        of << "### Classified samples: +EXPtestset" << endl;
        of << "Time per bootstrapping: +EXPbsEXACT" << endl;
        of << "Errors: +EXPnumERRabs / +EXPtestset (+EXPnumERRper)" << endl;
        of << "Disagreements: +EXPdis" << endl;
        of << "(pro-clear/pro-hom: +EXPclear / +EXPhom)" << endl;
        of << "Wrong bootstrappings: +EXPwrongBSabs" << endl;
        of << "Disagreements with wrong bootstrapping: +EXPwrongDISabs" << endl;
        of << "Avg. time for the evaluation of the network: +EXPtimeEXACT" << endl;
        of << "\\end{Verbatim}" << endl;
        of.close();
    }

    // free memory
    delete_gate_bootstrapping_secret_keyset(secret);
    delete_gate_bootstrapping_parameters(params);

    deleteTensor(weights,num_wire_layers, topology);
    deleteMatrix(biases, num_wire_layers);
    deleteMatrix(images, n_images);
    delete[] labels;

    return 0;
}


void deleteTensor(int*** tensor, int dim_tensor, const int* dim_vec)
{
    int** matrix;
    int dim_mat;
    for (int i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}


void deleteMatrix(int** matrix, int dim_mat)
{
    for (int i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

