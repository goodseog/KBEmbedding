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

const int dimension = 100;

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
vector<Triple> train, test;

void read_entity2vec() {
    cout << "Read entity2vec  ... ";

    FILE* fin = fopen( path_entity2vec.c_str(), "r");

    string ID;
    while( fscanf( fin, "%s", buf) > 0 ){
        ID = string(buf);
        vector<double> vec;
        for( int i = 0 ; i < dimension ; i++ ){
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
        for( int i = 0 ; i < dimension ; i++ ){
            double val;
            res_fscanf = fscanf(fin, "%lf", &val);
            vec.push_back(val);
        }
        relation2vec.push_back( kVector(ID, vec));
    }
    fclose(fin);
    cout << "done!" << endl;
}

void read_train(){
    cout << "Read train file   ... " ;
    FILE *fin = fopen(path_train2id.c_str(), "r");

    int train_case;    
    res_fscanf = fscanf(fin, "%d", &train_case);
    for( int tc = 1, h, l, t ; tc <= train_case ; tc++ ){
        res_fscanf = fscanf(fin, "%d", &h);
        res_fscanf = fscanf(fin, "%d", &t);
        res_fscanf = fscanf(fin, "%d", &l);
        train.push_back(Triple(h, l, t));
    }    
    fclose(fin);
    cout << "done!" << endl;
}

void read_test(){
    cout << "Read test file   ... " ;
    FILE *fin = fopen(path_test2id.c_str(), "r");
    
    int test_case;
    res_fscanf = fscanf(fin, "%d", &test_case);
    for( int tc = 1, h, l, t ; tc <= test_case ; tc++ ){
        res_fscanf = fscanf(fin, "%d", &h);
        res_fscanf = fscanf(fin, "%d", &t);
        res_fscanf = fscanf(fin, "%d", &l);        
        test.push_back(Triple(h, l, t));
    }    
    fclose(fin);
    cout << "done!" << endl;
}

void read_test_neg(){
    cout << "Read neg test file   ... " ;
    FILE *fin = fopen(path_test2id.c_str(), "r");
    
    int test_case;
    res_fscanf = fscanf(fin, "%d", &test_case);
    for( int tc = 1, h, l, t ; tc <= test_case ; tc++ ){
        res_fscanf = fscanf(fin, "%d", &h);
        res_fscanf = fscanf(fin, "%d", &t);
        res_fscanf = fscanf(fin, "%d", &l);        
        test.push_back(Triple(h, l, t));
    }    
    fclose(fin);
    cout << "done!" << endl;
}

vector<double> delta_l;

void preprocess(){
    cout << "Preprocess!! " << endl;
    delta_l.assign(relation2vec.size(), 0);

    cout << "num of Train case : " << train.size() << endl;

    vector<double> dist_vec(dimension);
    for( int i = 0 ; i < train.size() ; i++ ){
        auto &trp = train[i];

        auto &h_vec = entity2vec[trp.h].vec;
        auto &l_vec = relation2vec[trp.l].vec;
        auto &t_vec = entity2vec[trp.t].vec;

        for( int ii = 0 ; ii < dimension  ; ii++)
            dist_vec[ii] = h_vec[ii] + l_vec[ii] - t_vec[ii];
        
        if( delta_l[train[i].l] < vec_len(dist_vec) ){
            delta_l[train[i].l] = vec_len(dist_vec);
        }
    }
}

void link_prediction() {
    cout << "Link Prediction ... " << endl;
    cout << "num of Test case : " << test.size() << endl;

    int cnt_correct = 0, cnt_wrong = 0;

    vector< pair<double, int> > ranking(entity2vec.size());

    for( int i = 0 ; i < test.size() ; i++ ) {
        
        auto &trp = test[i];

        auto &h_vec = entity2vec[trp.h].vec;
        auto &l_vec = relation2vec[trp.l].vec;
        
                
        for( int j = 0 ; j < entity2vec.size() ; j++ ){
            double score = 0;
            auto &t_vec = entity2vec[j].vec;
            for( int k = 0 ; k < dimension ; k++)
                score += fabs(h_vec[k] + l_vec[k] - t_vec[k]);
            ranking[j] = {score, j};
        }

        sort(ranking.begin(), ranking.end());

        if( ranking[0].second == trp.t){
            FILE* fout = fopen( (string("link_prediction_FB13/") + to_string(i+1) + string(".txt")).c_str(), "w" );        

            fprintf(fout, "%s, %s, %s ?? \n", 
                entity2vec[trp.h].id.c_str(), 
                relation2vec[trp.l].id.c_str(), 
                entity2vec[trp.t].id.c_str() );

            for( int j = 0 ; j < 10 ; j++ )
                fprintf(fout, "%s %lf\n", 
                    entity2vec[ranking[j].second].id.c_str(), 
                    ranking[j].first);

            fclose(fout);
        }        
    }


}

void triple_classification() {
    
    cout << "Triple classification ... " << endl;
    cout << "num of Test case : " << test.size() << endl;

    int cnt_correct = 0, cnt_wrong = 0;

    vector<double> dist_vec(dimension);
    for( int i = 0 ; i < test.size() ; i++ ) {
        auto &trp = test[i];

        auto &h_vec = entity2vec[trp.h].vec;
        auto &l_vec = relation2vec[trp.l].vec;
        auto &t_vec = entity2vec[trp.t].vec;

        for( int j = 0 ; j < dimension ; j++)
            dist_vec[j] = h_vec[j] + l_vec[j] - t_vec[j];   

        if( vec_len(dist_vec) < delta_l[trp.l] ){
            // true
            cnt_correct++;
        }
        else {
            // false
            cnt_wrong++;
            cout << "Wrong : " << entity2vec[trp.h].id << "\t" << relation2vec[trp.l].id << "\t" << entity2vec[trp.t].id << endl;
        }
    }

    double accuracy = (double) cnt_correct / (cnt_correct + cnt_wrong);
    cout << "Accuracy : " << cnt_correct << " / " << (cnt_correct + cnt_wrong) << " ( "<< accuracy * 100.0 << "% )" << endl;
}

int main(){
    read_entity2vec();
    read_relation2vec();

    read_train();
    read_test();
    cout << "num of entity   : " << entity2vec.size() << endl;
    cout << "num of relation : " << relation2vec.size() << endl;
    cout << "num of test set : " << test.size() << endl;

    preprocess();

    for( int i = 0 ; i < delta_l.size() ; i++)
        cout << i << " " << delta_l[i] << endl;

    //link_prediction();
    triple_classification();

    

    return 0;
}