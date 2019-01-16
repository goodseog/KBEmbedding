#ifndef _MATHLIB_H
#define _MATHLIB_H

#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace std;

const double pi = 3.1415926535897932384626433832795;

// rand
// http://www.cplusplus.com/reference/cstdlib/rand/
// Returns a pseudo-random integral number in the range between 0 and RAND_MAX.
// min <= return value < max
double rand(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

double normal(double x, double miu, double sigma) {
    return 1.0 / sqrt(2 * pi) / sigma * exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}

double randn(double miu, double sigma, double min, double max) {
    double x, y, dScope;
    do {
        x = rand(min, max);
        y = normal(x, miu, sigma);
        dScope = rand(0.0, normal(miu, miu, sigma));
    } while (dScope > y);
    return x;
}

double sqr(double x) {
    return x * x;
}

double vec_len(vector<double> &a) {
    double res = 0;
    for (int i = 0; i < a.size(); i++)
        res += sqr(a[i]);
    res = sqrt(res);
    return res;
}

double normalize(vector<double> &a) {
    double x = vec_len(a);
    if (x > 1)
        for (int i = 0; i < a.size(); i++)
            a[i] /= x;
    return 0;
}

int rand_max(int x) {
    int res = (rand() * rand()) % x;
    while (res < 0)
        res += x;
    return res;
}

#endif