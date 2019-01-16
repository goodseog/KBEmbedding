#ifndef _CONFIGURE_H
#define _CONFIGURE_H

#include <string>
using namespace std;

// L1 norm (manhattan distance) vs L2 norm (square distance)
bool L1_flag = 1;

// training data path from 
// https://github.com/thunlp/OpenKE
string path_kb = "/home/goodseog/git/OpenKE/benchmarks/FB15K/";

string path_entity2id   = path_kb + "entity2id.txt";
string path_relation2id = path_kb + "relation2id.txt";
string path_train2id    = path_kb + "train2id.txt";
string path_test2id     = path_kb + "test2id.txt";

#endif