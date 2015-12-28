#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <list>
#include <map>
#include <tuple>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstring>
#include <thread>

#include "pool.h"

using namespace std;

/* stores data for a quadratic spline */
class Quadratic {
  public:

  /* store quadratic as "ax^2 + bx + c" */
  double a,b,c;

  /* start and end of spline */
  double L,R;
 
  /* book-keeping flag, set to true if the spline has been proven to have sufficient accuracy */
  bool accurate;
  
  /* fit to tuple of points */
  Quadratic(double x1, double y1, double x2, double y2, double x3, double y3) {
    L = x1;
    R = x3;
    a = ((y2 - y1)*(x1 - x3) + (y3 - y1)*(x2 - x1)) / ((x1-x3)*(x2*x2-x1*x1) + (x2-x1)*(x3*x3-x1*x1));
    b = ((y2 - y1) - a*(x2*x2 - x1*x1)) / (x2 - x1);
    c = y1 - a*x1*x1 - b*x1;

    accurate = false;
  }
  
  /* integrate with respect to gaussian with mean u and variance v */
  double Integrate(double u, double v)const {
    double t1, t2;
    double i2v = 0.5 / v;
    double sqrtv = sqrt(v);
    // sqrt(1.0 / (2Pi)) = 0.3989422804014327
    t1 = 0.3989422804014327 * sqrtv * (exp(-(L-u)*(L-u) * i2v) * (b + a*(L+u)) - exp(-(R-u)*(R-u) * i2v) * (b + a*(R+u)));
    t2 = 0.5 * (c + b*u + a*(u*u + v));
    // sqrt(0.5) = 0.7071067811865476
    double i_sqrt2v = 0.7071067811865476 / sqrtv;
    return t1 + t2*(erf((u - L) * i_sqrt2v) - erf((u - R) * i_sqrt2v));
  }
  
  /* return the value at x */
  inline double val(double x)const {
    return a * x * x + b*x + c;
  }

  /* return value at left */
  inline double left_val()const {
    return val(L);
  }
  /* return value at right */
  inline double right_val()const {
    return val(R);
  }
  /* return value at middle */
  inline double middle_val()const {
    return val((L + R)/2);
  }
};


/* a spline for our purposes is a list of piecewise quadratic functions and a linear asymptote for the right
hand side */
class Spline : public list<Quadratic> {
  public:
  /* stores the start of the linear asymptote */
  double R;
  /* stores the gradient */
  double a;
  /* stores the root of the spline */
  double L;
  /* stores the point at which the second arm becomes optimal */
  double divide;

  /* integrate all functions in the spline with respect to N(u, v) */
  double Integrate(double u, double v)const {
    /* this is for the linear asymptote */
    double sum = a * sqrt(v / (2 * M_PI)) * exp(-pow(R-u,2)/(2*v)) + 0.5*a*u*erfc((R-u)/sqrt(2*v));
    /* now iterate over quadratic pieces */
    for (auto q = begin();q!=end();++q) {
      sum+=q->Integrate(u, v); 
    }
    return sum;
  }

  double Value(double x) {
    if (x < L) {
      return 0.0;
    }
    if (x > R) {
      return a * x;
    }

    for (auto i = begin();i!=end();i++) {
      if (x >= i->L && x <= i->R) {
        return i->val(x);
      }
    }
    return 0;
  }
};

void ComputeIndex(map<vector<int>,Spline> *lookup, int n, int T1, int T2, double tolerance) {
  Spline next;
  if (n == 1) {
    next.R = 0.0;
    next.a = 1.0;
    (*lookup)[{n, T1, T2}] = next;
    return;
  }

  vector<int> e1 = {n-1,T1+1,T2};
  vector<int> e2 = {n-1,T1,T2+1};

  Spline V1 = (*lookup)[e1];
  Spline V2 = (*lookup)[e2];


  double v1 = 1.0 / (T1 * (T1 + 1.0));
  double v2 = 1.0 / (T2 * (T2 + 1.0));
  
  double left = -1.0;
  double right = 1.0;

  while (true) {
    double V = max(V1.Integrate(left, v1), left + V2.Integrate(left, v2));
    if (V < tolerance) {
      break;
    }
    left*=2;
  }

  while (true) {
    double V = max(V1.Integrate(right, v1), right + V2.Integrate(right, v2));
    if (V < right * n + tolerance) {
      break;
    }
    right*=2;
  }

  next.L = left;
  next.R = right;
  next.a = n;

  /* create the first quadratice spline on the interval [left,right] and matching [left,(left+right)/2,right] */
  double middle = (left + right) / 2.0;
  next.push_back(Quadratic(left, 0.0, middle, max(V1.Integrate(middle, v1), middle + V2.Integrate(middle, v2)), right, right * n));

  /* iterate over the splines, adding as the error is larger than the tolerance */
  auto i = next.begin();
  while (i != next.end()) {
    /* if the spline is known to be accurate then do nothing */
    if (!i->accurate) {
      /* otherwise calculate the two midpoints */
      double m = (i->L + i->R) * 0.5;
      double ml = (i->L + m) * 0.5;
      double mr = (i->R + m) * 0.5;
 
      /* calculate the values at the midpoints */
      double yl = max(V1.Integrate(ml, v1), ml + V2.Integrate(ml, v2));
      double yr = max(V1.Integrate(mr, v1), mr + V2.Integrate(mr, v2));
      
      /* and the errors with respect to the current spline */
      double errl = abs(yl - i->val(ml));
      double errr = abs(yr - i->val(mr));

      /* if the error is large enough, then split the quadratic into two */
      if (max(errl,errr) > tolerance) { 
        /* create two new quadratics to replace the old */
        Quadratic q1(i->L, i->left_val(), ml, yl, m, i->middle_val()); 
        Quadratic q2(m, i->middle_val(), mr, yr, i->R, i->right_val());
        
        /* if original spline was accurate in one half, then the new one should be too, so
        we don't need to check next time (saves some integrals) */
        q1.accurate = (errl < tolerance);
        q2.accurate = (errr < tolerance);
        
        /* erease the old spline and insert the new ones */
        i = next.erase(i);
        i = next.insert(i, q2);
        i = next.insert(i, q1);
        continue;
      }
    }
    ++i;
  }
  
  while (right - left > tolerance) {
    double m = (left + right) / 2.0;
    double vr = m + V2.Integrate(m, v2);
    double vl = V1.Integrate(m, v1);
    if (vr > vl) {
      right = m;
    }else {
      left = m;
    }
  }
  next.divide = (left + right) / 2.0;

  (*lookup)[{n,T1,T2}] = next;
}


void BuildTable(string fn, int n, double tolerance, int max_threads) {
  map<vector<int>,Spline> *lookup = new map<vector<int>,Spline>();
  for (int m = 1;m!=n+1;m++) {
    cout << "running at depth " << m << "\n";
    vector<thread> threads;
    for (int T1 = 1;T1!=n-m+2;T1++) {
      int T2 = 2 + n - m - T1;
      (*lookup)[{m,T1,T2}] = Spline();
    }
    for (int i = 0;i != max_threads;++i) {
      threads.push_back(std::thread([max_threads, i, lookup, m, n, tolerance] {
        for (int T1 = 1;T1!=n-m+2;T1++) {
          int T2 = 2 + n - m - T1;
          if (T1 % max_threads == i) {
            ComputeIndex(lookup, m, T1, T2, tolerance);
          }
        }
        return 0;
      })); 
    }
    for (int i = 0;i != max_threads;++i) {
      threads[i].join();
    }
    if (m > 1) {
      for (int T1 = 1;T1!=n-(m-1)+1;T1++) {
        int T2 = 2 + n - (m - 1) - T1;
        auto L = lookup->find({m-1,T1,T2});
        L->second.erase(L->second.begin(), L->second.end());
      }
    }
  }
  ofstream out(fn, ios::out | ios::binary);
  for (auto &e : *lookup) {
    out.write((char*)e.first.data(), sizeof(int) * 3);
    out.write((char*)&e.second.divide, sizeof(double));
  }
  out.close();
}



int main(int argc, char *argv[]) {
  if (argc <= 1) {
    goto die;
  }

  if (!strcmp(argv[1], "build") && argc == 6) {
    string fn = string(argv[2]);
    uint64_t n = atoi(argv[3]);
    double tolerance = atof(argv[4]);
    uint64_t max_threads = atoi(argv[5]);
    assert(n >= 1);
    assert(tolerance > 0);

    BuildTable(fn, n-2, tolerance, max_threads);
    return 0;
  }
die:
  cout << "Usage: makebayes build filename horizon tolerance max_threads\n";
  return 0;  
}

