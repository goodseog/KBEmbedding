#include <iostream>
#include <string>
#include <set>
#include <assert.h>

#include "mathlib.h"
#include "configure.h"

using namespace std;

char buf[100000];

// variables for "entity to integer"
int entity_num, relation_num;
map<string, int> entity2id, relation2id;
map<int, string> id2entity, id2relation;

// variables for bernoulli distribution training
map<int, map<int, int> > left_entity, right_entity;
map<int, double> left_num, right_num;

struct Train {
private:
    struct Triple {
        int h, l, t;
        Triple(){}
        Triple(int _h, int _l, int _t) : h(_h), l(_l), t(_t) {}
    };

    int k, method;
    double res;           // loss function value
    double count;         // loss function gradient
    double learning_rate, margin;
    vector<Triple> triples;
    vector<vector<double> > relation_vec, entity_vec;
    vector<vector<double> > relation_tmp, entity_tmp;
    map<pair<int, int>, map<int, int> > has_rel;

    void init_dimension() {
        // initialize vectors in k dimension
        relation_vec.resize(relation_num);
        for (int i = 0; i < relation_vec.size(); i++) 
            relation_vec[i].resize(k);
        relation_tmp.resize(relation_num);
        for (int i = 0; i < relation_tmp.size(); i++) 
            relation_tmp[i].resize(k);
        entity_vec.resize(entity_num);
        for (int i = 0; i < entity_vec.size(); i++)
            entity_vec[i].resize(k);
        entity_tmp.resize(entity_num);
        for (int i = 0; i < entity_tmp.size(); i++)
            entity_tmp[i].resize(k);
    }

    /*  void init_vector_values()
        1: initialize l <- uniform( - 6 / sqrt(k), 6 / sqrt(k)) for each l in L
        2:            l <- l / || l || for each l in L
        3:            e <- uniform( - 6 / sqrt(k), 6 / sqrt(k)) for each entity in E    
     */
    void init_vector_values() {
        
        for (int i = 0; i < relation_num; i++) {
            for (int ii = 0; ii < k; ii++)
                relation_vec[i][ii] = randn(0, 1.0 / k, -6 / sqrt(k), 6 / sqrt(k));
            normalize(relation_vec[i]);
        }

        for (int i = 0; i < entity_num; i++) {
            for (int ii = 0; ii < k; ii++)
                entity_vec[i][ii] = randn(0, 1.0 / k, -6 / sqrt(k), 6 / sqrt(k));
            normalize(entity_vec[i]);
        }            

    }

    void loop(){
        int nbatches = 100;
        int nepoch = 1000;
        int batchsize = triples.size() / nbatches;

        FILE* fscore = fopen( (path_res_output + "/cost.txt").c_str() , "w");
        // mini-batch training
        for (int epoch = 0; epoch < nepoch; epoch++) {
            res = 0;
            cout << "epoch #" << epoch << " : ";

            for (int batch = 0; batch < nbatches; batch++) {
                relation_tmp = relation_vec;
                entity_tmp = entity_vec;

                mini_batch_process(batchsize);

                relation_vec = relation_tmp;
                entity_vec = entity_tmp;
            }

            cout << res << endl;
            fprintf(fscore, "%.6lf\n", res);

            if( (epoch + 1) % 1000 == 0 )
                write_epoch_result_file(epoch);
            
        }
        fclose(fscore);
    }

    void mini_batch_process(int batchsize) {
        for (int iter = 0; iter < batchsize; iter++) {
            int i = rand_max(triples.size()); // select random triple index
            Triple &triple = triples[i];
            int j = rand_max(entity_num);  // select random entity index

            // bernoulli training
            double pr = 1000 * right_num[triple.l] / (right_num[triple.l] + left_num[triple.l]);
            
            if(method == 0) pr = 500; // uniform training

            if (rand() % 1000 < pr) { // train right entity
                while (has_rel[make_pair(triple.h, triple.l)].count(j) > 0)
                    j = rand_max(entity_num);
                train_kb(triple.h, triple.t, triple.l,
                         triple.h, j,        triple.l);
            } else { // train left entity
                while (has_rel[make_pair(j, triple.l)].count(triple.t) > 0)
                    j = rand_max(entity_num);
                train_kb(triple.h, triple.t, triple.l,
                         j,        triple.t, triple.l);
            }
            normalize(relation_tmp[triple.l]);
            normalize(entity_tmp[triple.h]);
            normalize(entity_tmp[triple.t]);
            normalize(entity_tmp[j]);
        }        
    }

    // 12: Update embeddings w.r.t.   Sum( gradient ( margin + d( h + l, t ) + d( h' + l, t') ))
    void train_kb( int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
        double sum1 = calc_score(e1_a, e2_a, rel_a);
        double sum2 = calc_score(e1_b, e2_b, rel_b);

        if ( margin + sum1 - sum2 > 0 ) {
            res += margin + sum1 - sum2;
            gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }

    double calc_score(int e1, int e2, int rel) {
        double score = 0;
        // L1_distance
        if (L1_flag)            
            for (int ii = 0; ii < k; ii++)
                score += fabs(entity_vec[e2][ii] - entity_vec[e1][ii] - relation_vec[rel][ii]);
        // L2_distance
        else            
            for (int ii = 0; ii < k; ii++)
                score += sqr(entity_vec[e2][ii] - entity_vec[e1][ii] - relation_vec[rel][ii]);
        return score;
    }


    void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
        for (int ii = 0; ii < k; ii++) {

            double x = 2 * (entity_vec[e2_a][ii] - entity_vec[e1_a][ii] - relation_vec[rel_a][ii]);            
            if (L1_flag)
                if (x > 0) x = 1;
                else       x = -1;
            relation_tmp[rel_a][ii] -= -1 * learning_rate * x;
            entity_tmp[e1_a][ii] -= -1 * learning_rate * x;
            entity_tmp[e2_a][ii] += -1 * learning_rate * x;

            x = 2 * (entity_vec[e2_b][ii] - entity_vec[e1_b][ii] - relation_vec[rel_b][ii]);
            if (L1_flag)
                if (x > 0) x = 1;
                else       x = -1;

            relation_tmp[rel_b][ii] -= learning_rate * x;
            entity_tmp[e1_b][ii] -= learning_rate * x;
            entity_tmp[e2_b][ii] += learning_rate * x;
        }
    }

    void write_epoch_result_file(int epoch_num) {
        string epoch_str = to_string(epoch_num);
        while(epoch_str.length() < 3) 
            epoch_str = "0" + epoch_str;

        FILE *f2 = fopen( path_relation2vec.c_str(), "w");
        for (int i = 0; i < relation_num; i++) {
            fprintf(f2, "%s\t", id2relation[i].c_str() );
            for (int ii = 0; ii < k; ii++)
                fprintf(f2, "%.6lf\t", relation_vec[i][ii]);
            fprintf(f2, "\n");
        }
        fclose(f2);

        FILE *f3 = fopen( path_entity2vec.c_str(), "w");
        for (int i = 0; i < entity_num; i++) {
            fprintf(f3, "%s\t", id2entity[i].c_str() );
            for (int ii = 0; ii < k; ii++)
                fprintf(f3, "%.6lf\t", entity_vec[i][ii]);
            fprintf(f3, "\n");
        }
        fclose(f3);
    }

public:
    // h --- l ---> t
    void add(int e_l, int e_r, int rel )
    {
        triples.push_back( Triple(e_l, rel, e_r) );
        has_rel[make_pair(e_l, rel)][e_r] = true;
    }

    void run(int dimension, double learning_rate, double margin){

        this->k = dimension;
        this->learning_rate = learning_rate;
        this->margin = margin;

        init_dimension();     // initialize vector dimension
        init_vector_values(); // line 1 - 3
        loop();               // line 4 - 13
    }

} train;

struct Preprocess
{
    void init_make_result_path (){
        cout << "mkdir result file      ...    " ;
        int res1 = system((string("mkdir ") + path_res_output).c_str());
        int res2 = system((string("rm ")    + path_res_output + "/*").c_str());
        cout << "done!" << endl;
    }

    void init_entity2id()
    {
        cout << "Read entity2id.txt      ...    ";
        FILE *f = fopen(path_entity2id.c_str(), "r");
        int x;

        int res = fscanf(f, "%d", &entity_num);
        for (int i = 0; i < entity_num; i++)
        {
            int res = fscanf(f, "%s%d", buf, &x);
            string st = buf;
            entity2id[st] = x;
            id2entity[x] = st;
        }

        fclose(f);
        cout << " done!" << endl;
    }

    void init_relation2id()
    {
        cout << "Read relation2id.txt    ...    ";
        FILE *f = fopen(path_relation2id.c_str(), "r");
        int x;

        int res = fscanf(f, "%d", &relation_num);
        for (int i = 0; i < relation_num; i++)
        {
            int res = fscanf(f, "%s%d", buf, &x);
            string st = buf;
            relation2id[st] = x;
            id2relation[x] = st;
        }

        fclose(f);
        cout << " done!" << endl;
    }

    void init_train()
    {
        cout << "Read train2id.txt       ...    ";
        FILE *fin_kb = fopen(path_train2id.c_str(), "r");

        int num_triples;
        int res = fscanf(fin_kb, "%d", &num_triples);

        for (int i = 0; i < num_triples; i++)
        {
            int entity_id_left, entity_id_right, relation_id;

            int res = fscanf(fin_kb, "%d%d%d", &entity_id_left, &entity_id_right, &relation_id);

            assert(entity_id_left < entity_num);
            assert(entity_id_right < entity_num);
            assert(relation_id < relation_num);

            left_entity[relation_id][entity_id_left]++;
            right_entity[relation_id][entity_id_right]++;
            train.add(entity_id_left, entity_id_right, relation_id);
        }

        fclose(fin_kb);
        cout << " done!" << endl;
    }

    void init_bernoulli() {
        for (int i = 0; i < relation_num; i++)
        {
            double sum1 = 0, sum2 = 0;
            for (map<int, int>::iterator it = left_entity[i].begin(); it != left_entity[i].end(); it++)
            {
                sum1++;
                sum2 += it->second;
            }
            left_num[i] = sum2 / sum1;
        }

        for (int i = 0; i < relation_num; i++)
        {
            double sum1 = 0, sum2 = 0;
            for (map<int, int>::iterator it = right_entity[i].begin(); it != right_entity[i].end(); it++)
            {
                sum1++;
                sum2 += it->second;
            }
            right_num[i] = sum2 / sum1;
        }
    }

    void run()
    {
        init_make_result_path();
        init_entity2id();
        init_relation2id();
        init_train();
        init_bernoulli();
        cout << "preprocess done!" << endl;
    }

} preprocess;



int main(int argc, char **argv)
{
    srand(time(NULL));
    int dimension = 100;
    double learning_rate = 0.001;
    double margin = 1;

    preprocess.run();
    train.run(dimension, learning_rate, margin);

    cout << "All process done!" << endl;

    return 0;
}