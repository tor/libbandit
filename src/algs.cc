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
#include "algs.h"
#include "bandit.h"
#include "arm.h"
#include "gittins_table.h"

#include <cfloat>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>

using namespace std;

/*************************************************************
UCB 
*************************************************************/
double alg_ucb(BanditProblem &bp, uint64_t n, double alpha) {
  bp.reset();
  uint64_t K = bp.K;
  vector<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max()));
  }


  uint64_t t = 0;
  for (;t != K && t !=n;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }

  for (;t != n;++t) {
    for (auto &a : arms) {
      a.idx = a.mean() + sqrt(alpha / a.T * log((t+1.0)));
    }
    auto best = max_element(arms.begin(), arms.end());
    best->pull(bp.choose(best->i));
  }
  return bp.get_regret();
}


/*************************************************************
OCUCB
*************************************************************/
double alg_ocucb(BanditProblem &bp, uint64_t n, double alpha, double psi) {
  bp.reset();
  uint64_t K = bp.K;
  vector<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }

  uint64_t t = 0;
  for (;t!=n && t!=K;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }
  bool update = true;
  for (;t != n;++t) {
    if (update) {
      for (auto &a : arms) {
        a.idx = a.mean() + sqrt(alpha / a.T * log(psi * n / (double)(t+1)));
      }
      sort(arms.rbegin(), arms.rend());
      update = false;
    }
    arms[0].pull(bp.choose(arms[0].i));
    if (arms[0].mean() + sqrt(alpha / arms[0].T * log(psi * n / (double)(t+1))) < arms[1].idx) {
      update = true;
    }
  }
  return bp.get_regret();
}


/*************************************************************
AOUCB
*************************************************************/
double alg_aocucb(BanditProblem &bp, uint64_t n, double alpha) {
  bp.reset();
  int K = bp.K;
  priority_queue<Arm> arms;

  for (int i = 0;i != K;++i) {
    arms.push(Arm(i, numeric_limits<double>::max())); 
  }

  for (uint64_t t = 0;t != n;++t) {
    auto best = arms.top();
    arms.pop();
    best.pull(bp.choose(best.i));
    best.idx = best.mean() + sqrt(alpha / best.T * log((double)n / best.T));
    arms.push(best);
  }
  return bp.get_regret();
}

/*************************************************************
MOSS
*************************************************************/
double alg_moss(BanditProblem &bp, uint64_t n, double alpha) {
  bp.reset();
  int K = bp.K;
  priority_queue<Arm> arms;

  for (int i = 0;i != K;++i) {
    arms.push(Arm(i, numeric_limits<double>::max())); 
  }

  for (uint64_t t = 0;t != n;++t) {
    auto best = arms.top();
    arms.pop();
    best.pull(bp.choose(best.i));
    best.idx = best.mean() + sqrt(alpha / best.T * log(max(1.0, (double)n / (K * best.T))));
    arms.push(best);
  }
  return bp.get_regret();
}

/*************************************************************
GITTINS APPROXIMATION
*************************************************************/
double alg_gaussian_gittins_approx(BanditProblem &bp, uint64_t n) {
  bp.reset();
  uint64_t K = bp.K;
  vector<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max()));
  }


  uint64_t t = 0;
  for (;t != K && t !=n;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }

  for (;t != n;++t) {
    for (auto &a : arms) {
      uint64_t m = n - t;
      double beta = max(1.0, min(m / pow(log(m), 1.5) / 4.0, m / 4.0 / a.T / pow(log(m/a.T), 0.5)));
      a.idx = a.mean() + sqrt(2.0 / a.T * log(beta));
    }
    auto best = max_element(arms.begin(), arms.end());
    best->pull(bp.choose(best->i));
  }
  return bp.get_regret();
}


/*************************************************************
GITTINS
*************************************************************/
double alg_gaussian_gittins(BanditProblem &bp, uint64_t n, GittinsTable &table) {
  bp.reset();
  uint64_t K = bp.K;
  vector<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, 0.0));
  }
  
  uint64_t t = 0;
  for (;t != K && t!=n;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }
  bool update = true;
  for (;t != n;++t) {
    if (update) {
      for (auto &a : arms) {
        a.idx = a.mean() + table.get_idx(n - t, a.T);
      }
      sort(arms.rbegin(), arms.rend());
      update = false;
    }
    arms[0].pull(bp.choose(arms[0].i));
    if (arms[0].mean() + table.get_idx(n - t, arms[0].T) < arms[1].idx) {
      update = true;
    }
  }
  return bp.get_regret();
}


/*************************************************************
THOMPSON SAMPLING
*************************************************************/
double alg_gaussian_ts(BanditProblem &bp, uint64_t n, default_random_engine &gen) {
  bp.reset();
  int K = bp.K;
  vector<Arm> arms;

  normal_distribution<double> dist(0.0, 1.0);

  for (int i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }

  uint64_t t = 0;

  for (;t != (uint64_t)K && t != n;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }

  for (;t != n;++t) {
    for (auto &a : arms) {
      a.idx = a.mean() + dist(gen) / sqrt(1 + a.T);
    }
    auto best = max_element(arms.begin(), arms.end());
    best->pull(bp.choose(best->i));
  }
  return bp.get_regret();
}




/*************************************************************
CONSERVATIVE UCB
*************************************************************/
double alg_conservative_ucb(BanditProblem &bp, uint64_t n, double alpha, double delta) {
  bp.reset();
  double mu0 = bp.mean(0);
  uint64_t K = bp.K;
  vector<Arm> arms;

  vector<double> lambda(K);

  for (int i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }
  double x = log(K / delta); 
  double y = log(max(3.0, log(K / delta))) + log(2.0 * M_E * M_E * K / delta);


  for (uint64_t t = 0;t != n;++t) {
    double Z = 0.0;
    for (int i = 0;i != K;++i) { 
      if (i == 0) {
        arms[i].idx = mu0;
        lambda[i] = mu0;
      }else {
        if (arms[i].T == 0) {
          arms[i].idx = numeric_limits<double>::max();
          lambda[i] = 0.0;
        }else {
          double psi = x*(1.0 + log(x)) / ((x - 1.0) * log(x)) * log(log(arms[i].T+1)) + y;
          arms[i].idx = arms[i].mean() + sqrt(2.0 / arms[i].T * psi);
          lambda[i] = max(0.0, arms[i].mean() - sqrt(2.0 / arms[i].T * psi));
        }
      }
      Z+=lambda[i] * arms[i].T;
    }
    auto J = max_element(arms.begin(), arms.end());
    Z += lambda[J->i];
    Z -= (1.0 - alpha) * mu0 * (t+1);
    
    if (Z >= 0) {
      J->pull(bp.choose(J->i));
    }else {
      arms[0].pull(bp.choose(0));
    }
  }
  return bp.get_regret();
}



/*************************************************************
UNBALANCED MOSS
*************************************************************/
double alg_unbalanced_moss(BanditProblem &bp, uint64_t n, vector<double> B) {
  bp.reset();
  uint64_t K = bp.K;
  priority_queue<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push(Arm(i, numeric_limits<double>::max())); 
  }

  for (uint64_t t = 0;t != n;++t) {
    auto best = arms.top();
    arms.pop();
    best.pull(bp.choose(best.i));
    best.idx = best.mean() + sqrt(4.0 / best.T * log(max(1.0, (double)(n * n) / (B[best.i] * B[best.i] * best.T))));
    arms.push(best);
  }
  return bp.get_regret();
}
