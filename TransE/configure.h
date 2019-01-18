#ifndef _CONFIGURE_H
#define _CONFIGURE_H

#include <string>
using namespace std;

// L1 norm (manhattan distance) vs L2 norm (square distance)
bool L1_flag = 1;

// training data path from 
// https://github.com/thunlp/OpenKE
string path_db = "/home/goodseog/git/OpenKE/benchmarks";
string dataset = "FB13";

string path_entity2id   = path_db + "/" + dataset + "/" + "entity2id.txt";
string path_relation2id = path_db + "/" + dataset + "/" + "relation2id.txt";
string path_train2id    = path_db + "/" + dataset + "/" + "train2id.txt";
string path_test2id     = path_db + "/" + dataset + "/" + "test2id.txt";

string path_res_output = "result_" + dataset;

string path_entity2vec   = "./result_" + dataset + "/entity2vec.txt";
string path_relation2vec = "./result_" + dataset + "/relation2vec.txt";

#endif