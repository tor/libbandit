/***************************************************************************
LibBandit - Multi-Armed Bandit Library
Written in 2015 by Tor Lattimore tor.lattimore@gmail.com

To the extent possible under law, the author(s) have dedicated all 
copyright and related and neighboring rights to this software to the 
public domain worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication 
along with this software. If not, 
see http://creativecommons.org/publicdomain/zero/1.0/
***************************************************************************/


/************************************************************************** 
Code for computing Gittins indices with Gaussian prior/noise model.
Author: Tor Lattimore, 2015
**************************************************************************/
#include <iostream>
#include <list>
#include <limits>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "pool.h"
#include "gittins_table.h"

#define MAX_IDX 5

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
};

/* This finds the root of the new spline from the old */
double FindRoot(const Spline &prev, double var, double tolerance) {
  double l = -MAX_IDX;
  double u = prev.L;
  while (u - l > tolerance) {
    double m = (u + l) / 2;
    double y = m + prev.Integrate(m, var);
    if (y <= 0) {
      l = m;
    }else {
      u = m;
    }
  }
  return (u + l) / 2.0;
}

/* compute bellman backup */
void Backup(const Spline &prev, Spline &next, int t, double var, double tolerance) {

  /* find the gittins index, which is the root of the integral of the splines in prev */
  double left = FindRoot(prev, var, tolerance);

  /* find an estimate of where it is safe to start the right asymptote */
  double right = 1.0;

  while (abs(right * t - right - prev.Integrate(right, var)) > tolerance) {
    right *= 2;
  }
  next.R = right;
  next.a = t;

  /* create the first quadratice spline on the interval [left,right] and matching [left,(left+right)/2,right] */
  double middle = (left + right) / 2.0;
  next.push_back(Quadratic(left, left + prev.Integrate(left, var), middle, middle + prev.Integrate(middle, var), right, right * t));

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
      double yl = ml + prev.Integrate(ml, var);
      double yr = mr + prev.Integrate(mr, var);
   
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
  next.L = next.begin()->L;
}

/* compute the index */
void ComputeIndex(int n, int T, double tolerance, GittinsTable *table) {
  /* the first value function is just the hinge function */ 
  Spline first;
  first.L = 0.0;
  first.a = 1.0;
  first.R = 0.0;


  /* iterate starting from the end and backup the value function and push the index */
  for (int t = 2; t!= n+1;++t) {
    Spline next;

    /* compute the inverse variance of the posterior */
    uint64_t Tn = (T + n - t);

    /* compute the variance at this level */
    double var = 1.0 / (Tn * (Tn+1));

    /* backup */
    Backup(first, next, t, var, tolerance);

    /* add the gittins index */
    if (table != nullptr) {
      table->set_idx(t, Tn, -next.begin()->L);
    }else if (t == n) {
      cout << "index for (" << t << ", " << Tn << ") is " << -next.begin()->L << " based on " << next.size() << " splines\n";
    }
    first = next;
  }
}

/* accepts a filename, horizon, tolerance and number of threads. Writes
a table of indices */
void BuildTable(string fn, int n, double tolerance, int max_threads) {
  Pool<int> pool(max_threads);  
  GittinsTable *table = new GittinsTable(n);
  for (int t = n;t!=0;--t) {
    pool.push([t, tolerance,table]{ComputeIndex(t, 1, tolerance, table); return 0;});
  }
  pool.run();
  
  table->write(fn);
}


int main(int argc, char *argv[]) {
  if (argc <= 1) {
    goto die;
  }
  
  if (!strcmp(argv[1], "lookup") && argc == 5) {
    string fn = string(argv[2]);
    uint64_t n = atoi(argv[3]);
    uint64_t T = atoi(argv[4]);
    
    assert(n >= 1);
    assert(T >= 1);

    GittinsTable table(fn);
    cout << table.get_idx(n, T) << "\n";
    return 0;
  }

  if (!strcmp(argv[1], "build") && argc == 6) {
    string fn = string(argv[2]);
    uint64_t n = atoi(argv[3]);
    double tolerance = atof(argv[4]);
    uint64_t max_threads = atoi(argv[5]);

    assert(n >= 1);
    assert(max_threads >= 1);
    assert(tolerance > 0);

    BuildTable(fn, n, tolerance, max_threads);
    return 0;
  }

  if (!strcmp(argv[1], "compute") && argc == 5) {
    uint64_t m = atoi(argv[2]);
    uint64_t T = atoi(argv[3]);
    double tolerance = atof(argv[4]);

    assert(m >= 1);
    assert(T >= 1);
    assert(tolerance > 0);

    ComputeIndex(m, T, tolerance, nullptr);
    return 0;
  }

die:
  cout << "Usage: makegittins build filename horizon tolerance maxthreads\n";
  cout << "   or: makegittins lookup filename n T\n";
  cout << "   or: makegittins compute n T tolerance\n";
  return 0;
}





