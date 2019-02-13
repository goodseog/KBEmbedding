#ifndef _CONFIGURE_H
#define _CONFIGURE_H

#include <string>
using namespace std;

// L1 norm (manhattan distance) vs L2 norm (square distance)
bool L1_flag = 1;

// training data path from 
// https://github.com/thunlp/OpenKEs
string path_db = "/home/goodseog/git/OpenKE/benchmarks";
string dataset = "FB15K";

string path_entity2id   = path_db + "/" + dataset + "/" + "entity2id.txt";
string path_relation2id = path_db + "/" + dataset + "/" + "relation2id.txt";
string path_train2id    = path_db + "/" + dataset + "/" + "train2id.txt";
string path_test2id     = path_db + "/" + dataset + "/" + "test2id.txt";

string path_res_output = "../result";

string file_entity2vec   = "entity2vec.txt";
string file_relation2vec = "relation2vec.txt";
string file_cost         = "cost.txt";

#endif