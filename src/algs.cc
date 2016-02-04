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
#include <list>

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

  for (auto &a : arms) {
    a.idx = a.mean() + sqrt(alpha / a.T * log(psi * n / (double)(K+1)));
  }
  sort(arms.begin(), arms.end(), greater<Arm>());
  bool update = false;
  for (;t != n;++t) {
    double bonus = alpha * log(psi * n / (double)(t+1));
    double idx = arms[0].mean() + sqrt(bonus / arms[0].T);

    if (idx < arms[1].idx) {
      update = true;
    }

    if (update) {
      arms[0].idx = idx;
      for (int i = 1;i != K;++i) {
        arms[i].idx = arms[i].mean() + sqrt(bonus / arms[i].T);
      }
      sort(arms.begin(), arms.end(), greater<Arm>());
      update = false;
    }
    arms[0].pull(bp.choose(arms[0].i));
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
  uint64_t K = bp.K;
  list<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }

  uint64_t t = 0;
  for (auto &a : arms) {
    a.pull(bp.choose(a.i));
    a.idx = a.mean() + sqrt(alpha * log(max(1.0, (double)n / K)));
  }

  arms.sort(greater<Arm>());

  for (;t != n;++t) {
    double idx = arms.begin()->mean() + sqrt(alpha / arms.begin()->T * log(max(1.0, (double)n / (K * arms.begin()->T))));
    arms.begin()->idx = idx;
    
    auto i = arms.begin();
    i++;

    while (i != arms.end() && idx < i->idx) {
      i++;
    }
    
    arms.splice(i, arms, arms.begin());
    arms.begin()->pull(bp.choose(arms.begin()->i));
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
    int max_index = -1;
    double m = -numeric_limits<double>::max();
    for (int i = 0;i != K;++i) {
      double idx = arms[i].mean() + dist(gen) / sqrt(arms[i].T);
      if (idx > m) {
        m = idx;
        max_index = i;
      }
    }
    arms[max_index].pull(bp.choose(arms[max_index].i));
  }
  return bp.get_regret();
}

/*************************************************************
BUDGETFIRST UCB
*************************************************************/
double alg_budget_first(BanditProblem &bp, uint64_t n, double alpha, double delta) {
  bp.reset();
  uint64_t K = bp.K;
  double mu0 = bp.mean(0);
  double x = log(K / delta); 
  double y = log(max(3.0, log(K / delta))) + log(2.0 * M_E * M_E * K / delta);
  double max_regret = sqrt(2.0 *n*(K-1) * (x*(1.0 + log(x)) / ((x - 1.0) * log(x)) * log(log(n+1)) + y));
  double t0 = min((double)n, max_regret / (alpha * mu0));
  vector<Arm> arms;
  for (int i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }
  uint64_t t = 0;
  for (;t < t0 && t < n;++t) {
    bp.choose(0);  
  }
  for (;t<n;++t) {
    auto J = max_element(arms.begin(), arms.end());
    J->pull(bp.choose(J->i));
    double psi = x*(1.0 + log(x)) / ((x - 1.0) * log(x)) * log(log(J->T+1)) + y;
    J->idx = J->mean() + sqrt(2.0 / J->T * psi);
  }
  return bp.get_regret();
}


/*************************************************************
CONSERVATIVE UCB
*************************************************************/
double alg_conservative_ucb(BanditProblem &bp, uint64_t n, double alpha, double delta, bool mu_known) {
  bp.reset();
  double mu0 = 0.0;
  if (mu_known) {
    mu0 = bp.mean(0);
  }
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
      if (i == 0 && mu_known) {
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
      if (i != 0) {
        Z+=lambda[i] * arms[i].T;
      }
    }
    auto J = max_element(arms.begin(), arms.end());

    Z += lambda[J->i];

    if (mu_known) {
      Z += arms[0].T * mu0 - (1.0 - alpha) * mu0 * (t+1);
    }else {
      Z += arms[0].T * arms[0].idx - (1.0 - alpha) * arms[0].idx * (t+1);
    }
    
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


/*************************************************************
GAUSSIAN BAYES (TWO ARMS ONLY)
*************************************************************/
double alg_gaussian_bayes(BanditProblem &bp, uint64_t n, BayesTable &table) {
  bp.reset();
  uint64_t K = bp.K;
  assert(K = 2);
  vector<Arm> arms;

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, numeric_limits<double>::max())); 
  }
  for (uint64_t i = 0;i != 2;++i) {
    arms[i].pull(bp.choose(i));
  }

  for (uint64_t t = 2;t != n;++t) {
    double delta = arms[1].mean() - arms[0].mean();
    double divide = table.lookup({n-t, arms[0].T, arms[1].T});

    if (delta > divide) {
      arms[1].pull(bp.choose(1));
    }else {
      arms[0].pull(bp.choose(0));
    }
  }
  return bp.get_regret();
}




