#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <queue>
#include <limits>
#include <cstddef>
#include <cmath>

#include "configure.h"
#include "mathlib.h"

using namespace std;

int k = 100;
int test_case;


int res_fscanf;
char buf[1000];

struct kVector {
    string id;
    vector<double> vec;
    kVector(){}
    kVector(string &_id, vector<double> &_vec){
        id = _id;
        vec = _vec;
    }
};

struct Triple{
    int h, l, t;
    Triple(){}
    Triple( int h, int l, int t) {
        this->h = h;
        this->l = l;
        this->t = t;
    }
};

vector<kVector> entity2vec, relation2vec;
vector<Triple> test;

void read_entity2vec() {
    cout << "Read entity2vec  ... ";

    FILE* fin = fopen( path_entity2vec.c_str(), "r");

    string ID;
    while( fscanf( fin, "%s", buf) > 0 ){
        ID = string(buf);
        vector<double> vec;
        for( int i = 0 ; i < k ; i++ ){
            double val;
            res_fscanf = fscanf(fin, "%lf", &val);
            vec.push_back(val);
        }
        entity2vec.push_back( kVector(ID, vec));
    }
    fclose(fin);
    cout << "done!" << endl;
}

void read_relation2vec() {
    cout << "Read relation2vec ... ";    
    FILE* fin = fopen( path_relation2vec.c_str(), "r");

    string ID;
    while( fscanf( fin, "%s", buf) > 0 ){
        ID = string(buf);
        vector<double> vec;
        for( int i = 0 ; i < k ; i++ ){
            double val;
            res_fscanf = fscanf(fin, "%lf", &val);
            vec.push_back(val);
        }
        relation2vec.push_back( kVector(ID, vec));
    }
    fclose(fin);
    cout << "done!" << endl;
}

void read_test(){
    cout << "Read test file   ... " ;
    FILE *fin = fopen(path_test2id.c_str(), "r");
    
    res_fscanf = fscanf(fin, "%d", &test_case);
    for( int tc = 1, h, l, t ; tc <= test_case ; tc++ ){
        res_fscanf = fscanf(fin, "%d", &h);
        res_fscanf = fscanf(fin, "%d", &l);
        res_fscanf = fscanf(fin, "%d", &t);
        test.push_back(Triple(h, l, t));
    }    
    fclose(fin);
    cout << "done!" << endl;
}

vector<double> delta_l;

void preprocess(){
    cout << "Preprocess!! " << endl;
    delta_l.assign(relation2vec.size(), 0);

    FILE *fin = fopen( path_train2id.c_str(), "r");
    int train_case;
    int res = fscanf(fin, "%d", &train_case);

    cout << "num of Train case : " << train_case << endl;

    vector<double> dist_vec;
    dist_vec.resize(k);
    for( int i = 0 ; i < train_case ; i++ ){
        cout << i << " ";

        int h, l, t;
        res = fscanf(fin, "%d", &h);
        res = fscanf(fin, "%d", &l);
        res = fscanf(fin, "%d", &t);
        
        cout << h << " " << l << " " << t;
        auto& h_vec = entity2vec[h].vec;
        auto& l_vec = relation2vec[l].vec;
        auto& t_vec = entity2vec[t].vec;

        cout << " -- ";

        for( int ii = 0 ; ii < k ; ii++)
            dist_vec[ii] = h_vec[ii] + l_vec[ii] - t_vec[ii];

        delta_l[l] = max( delta_l[l], vec_len(dist_vec));
        cout << " ## ";
    }

    fclose(fin);
}

void triple_classification() {
    

}

int main(){
    read_entity2vec();
    read_relation2vec();
    read_test();
    cout << "num of entity   : " << entity2vec.size() << endl;
    cout << "num of relation : " << relation2vec.size() << endl;
    cout << "num of test set : " << test.size() << endl;

    preprocess();

    for( int i = 0 ; i < delta_l.size() ; i++)
        cout << i << " " << delta_l[i] << endl;

    // link_prediction();
    triple_classification();

    

    return 0;
}