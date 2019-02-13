#include <iostream>
#include <string>
#include <set>
#include <assert.h>

#include "../include/mathlib.h"
#include "../include/configure.h"

using namespace std;

string algorithm = "TransH";
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
    vector<vector<double> > A, A_tmp;
    vector<vector<double> > relation_vec, entity_vec;
    vector<vector<double> > relation_tmp, entity_tmp;
    map<pair<int, int>, map<int, int> > has_rel;

    double normalize_aA(vector<double> &a, vector<double> &A)  {
        normalize2one(A);
        double sum = 0;

        while (true) {
            for (int i = 0; i < k; i++) 
                sum += sqr(A[i]);
            sum = sqrt(sum);

            for (int i = 0; i < k; i++) 
                A[i] /= sum;

            double x = 0;
            for (int ii = 0; ii < k; ii++)
                x += A[ii] * a[ii];

            if (x > 0.1) {
                for (int ii = 0; ii < k; ii++) {
                    a[ii] -= learning_rate * A[ii];
                    A[ii] -= learning_rate * a[ii];
                }
            }
            else
                break;
        }
        normalize2one(A);
        return 0;
    }

    void init_dimension() {
        // initialize A;
        A.resize(relation_num);
		for (int i=0; i<relation_num; i++)
            A[i].resize(k);

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
        for (int i=0; i<relation_num; i++) {
		    for (int j=0; j < k; j++)
		    	A[i][j] = randn(0 , 1.0 / k, -1, 1);
		    normalize2one(A[i]);
		}

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

        string path_cost = path_res_output + "/" + algorithm + "/" + dataset + "/" + file_cost;
        FILE* fscore = fopen( path_cost.c_str() , "w");
        // mini-batch training
        for (int epoch = 0; epoch < nepoch; epoch++) {
            res = 0;
            cout << "epoch #" << epoch << " : ";

            for (int batch = 0; batch < nbatches; batch++) {
                A_tmp = A;
                relation_tmp = relation_vec;
                entity_tmp = entity_vec;

                mini_batch_process(batchsize);

                A = A_tmp;
                relation_vec = relation_tmp;
                entity_vec = entity_tmp;
            }

            cout << res << endl;
            fprintf(fscore, "%.6lf\n", res);

            if( epoch == nepoch - 1 ){
                cout << "write result file" << endl;
                write_epoch_result_file(epoch);
            }
                
            
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
            normalize(entity_tmp[triple.h]);
            normalize(entity_tmp[triple.t]);
            normalize(entity_tmp[j]);

            normalize_aA(entity_tmp[triple.h],A_tmp[triple.l]);
            normalize_aA(entity_tmp[triple.t],A_tmp[triple.l]);
            normalize_aA(entity_tmp[j],A_tmp[triple.l]);
        }
    }

    void train_kb( int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
        double sum1 = calc_score(e1_a, e2_a, rel_a);
        double sum2 = calc_score(e1_b, e2_b, rel_b);

        if ( margin + sum1 - sum2 > 0 ) {
            res += margin + sum1 - sum2;
        	gradient(e1_a, e2_a, rel_a, -1);
        	gradient(e1_b, e2_b, rel_b, +1);
        }
    }

    double calc_score(int e1, int e2, int rel) {
        double tmp1=0, tmp2=0;
        for (int i = 0; i < k; i++) {
        	tmp1 += A[rel][i]*entity_vec[e1][i];
            tmp2 += A[rel][i]*entity_vec[e2][i];
        }

        double score=0;
        for (int i = 0; i < k; i++)
            score += fabs(entity_vec[e2][i] - tmp2 * A[rel][i] - (entity_vec[e1][i] - tmp1 * A[rel][i]) - relation_vec[rel][i]);
        return score;
    }

    void gradient(int e1, int e2, int rel, double belta) {
        double tmp1 = 0, tmp2 = 0;
        double sum_x = 0;
        for (int i = 0; i < k; i++) {
            tmp1 += A[rel][i] * entity_vec[e1][i];
            tmp2 += A[rel][i] * entity_vec[e2][i];
        }

        for (int i = 0; i < k; i++) {
            double x = 2 * (entity_vec[e2][i] - tmp2 * A[rel][i] - (entity_vec[e1][i] - tmp1 * A[rel][i]) - relation_vec[rel][i]);
            //for L1 distance function            
            if (x > 0) x = +1;
            else       x = -1;

            sum_x += x * A[rel][i];
            relation_tmp[rel][i] -= belta * learning_rate * x;
            entity_tmp[e1][i] -= belta * learning_rate * x;
            entity_tmp[e2][i] += belta * learning_rate * x;
            A_tmp[rel][i] += belta * learning_rate * x * tmp1;
            A_tmp[rel][i] -= belta * learning_rate * x * tmp2;
        }

        for (int i = 0; i < k; i++) {
            A_tmp[rel][i] += belta * learning_rate * sum_x * entity_vec[e1][i];
            A_tmp[rel][i] -= belta * learning_rate * sum_x * entity_vec[e2][i];
        }

        normalize(relation_tmp[rel]);
        normalize(entity_tmp[e1]);
        normalize(entity_tmp[e2]);

        normalize2one(A_tmp[rel]);
        normalize_aA(relation_tmp[rel], A_tmp[rel]);
    }

    void write_epoch_result_file(int epoch_num) {
        string epoch_str = to_string(epoch_num);
        while(epoch_str.length() < 3) 
            epoch_str = "0" + epoch_str;


        string path_relation2vec = path_res_output + "/" + algorithm + "/" + dataset + "/" + file_relation2vec;
        FILE *f2 = fopen( path_relation2vec.c_str(), "w");
        for (int i = 0; i < relation_num; i++) {
            fprintf(f2, "%s\t", id2relation[i].c_str() );
            for (int ii = 0; ii < k; ii++)
                fprintf(f2, "%.6lf\t", relation_vec[i][ii]);
            fprintf(f2, "\n");
        }
        fclose(f2);

        string path_entity2vec = path_res_output + "/"  + algorithm + "/" + dataset + "/" + file_entity2vec;
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
    void add(int e_l, int e_r, int rel ) {
        triples.push_back( Triple(e_l, rel, e_r) );
        has_rel[make_pair(e_l, rel)][e_r] = true;
    }

    void run(int dimension, double learning_rate, double margin) {
        this->k = dimension;
        this->learning_rate = learning_rate;
        this->margin = margin;

        init_dimension();     // initialize vector dimension
        init_vector_values(); // line 1 - 3
        loop();               // line 4 - 13
    }
} train;

struct Preprocess {
    void init_make_result_path () {
        cout << "mkdir result file      ...    " ;
        int res1 = system((string("mkdir ") + path_res_output + "/"  + algorithm ).c_str());
        int res2 = system((string("mkdir ") + path_res_output + "/"  + algorithm + "/" + dataset ).c_str());
        cout << "done!" << endl;
    }

    void init_entity2id() {
        cout << "Read entity2id.txt      ...    ";
        FILE *f = fopen(path_entity2id.c_str(), "r");
        int x;

        int res = fscanf(f, "%d", &entity_num);
        for (int i = 0; i < entity_num; i++) {
            int res = fscanf(f, "%s%d", buf, &x);
            string st = buf;
            entity2id[st] = x;
            id2entity[x] = st;
        }

        fclose(f);
        cout << " done!" << endl;
    }

    void init_relation2id() {
        cout << "Read relation2id.txt    ...    ";
        FILE *f = fopen(path_relation2id.c_str(), "r");
        int x;

        int res = fscanf(f, "%d", &relation_num);
        for (int i = 0; i < relation_num; i++) {
            int res = fscanf(f, "%s%d", buf, &x);
            string st = buf;
            relation2id[st] = x;
            id2relation[x] = st;
        }

        fclose(f);
        cout << " done!" << endl;
    }

    void init_train() {
        cout << "Read train2id.txt       ...    ";
        FILE *fin_kb = fopen(path_train2id.c_str(), "r");

        int num_triples;
        int res = fscanf(fin_kb, "%d", &num_triples);

        for (int i = 0; i < num_triples; i++) {
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
        for (int i = 0; i < relation_num; i++) {
            double sum1 = 0, sum2 = 0;
            for (map<int, int>::iterator it = left_entity[i].begin(); it != left_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            left_num[i] = sum2 / sum1;
        }

        for (int i = 0; i < relation_num; i++) {
            double sum1 = 0, sum2 = 0;
            for (map<int, int>::iterator it = right_entity[i].begin(); it != right_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            right_num[i] = sum2 / sum1;
        }
    }

    void run() {
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