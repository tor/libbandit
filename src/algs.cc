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

#include <boost/math/special_functions/erf.hpp>

using namespace std;

/*************************************************************
GENERIC SIMULATOR
*************************************************************/
double IndexAlgorithm::sim(BanditProblem &bp, uint64_t horizon) {
  K = bp.K;
  n = horizon;

  bp.reset();
  arms.clear();

  for (uint64_t i = 0;i != K;++i) {
    arms.push_back(Arm(i, std::numeric_limits<double>::max()));
    arms.rbegin()->max_idx = std::numeric_limits<double>::max();
  }

  uint64_t t = 0;
  for (;t != K && t !=n;++t) {
    arms[t].pull(bp.choose(arms[t].i));
  }
  
  for (;t != n;++t) {
    auto best_idx = -numeric_limits<double>::max();
    auto best = arms.begin();
    auto last = arms.begin();

    for (auto a = arms.begin();a!=arms.end();++a) {
      if (a->max_idx < best_idx) {
        break;
      }
      last = a;
      set_index(a, t);

      if (a->idx > best_idx) {
        best_idx = a->idx;
        best = a;
      }
    }
    best->max_idx = numeric_limits<double>::max();
    best->pull(bp.choose(best->i));

    update(best);

    if (best != arms.begin()) {
      sort(arms.begin(), last+1, [](const Arm &a1, const Arm &a2) {return a1.max_idx > a2.max_idx;});
      inplace_merge(arms.begin(), last+1, arms.end(), [](const Arm &a1, const Arm &a2) {return a1.max_idx > a2.max_idx;});
    }
  }
  return bp.get_regret();
}


void UCB::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + sqrt(2.0 / a->T * log(t));
  a->max_idx = a->mean() + sqrt(2.0 / a->T * log(n));
}

void MOSS::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + sqrt(2.0 / a->T * log(max(1.0, (double)n / (a->T * K))));
  a->max_idx = a->idx;
}

void OCUCB::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + sqrt(3.0 / a->T * log(2.0 * (double)n / t));
  a->max_idx = a->idx;
}

void AnytimeOCUCB::set_index(vector<Arm>::iterator a, uint64_t t) {
  const double EULER = exp(1.0);
  a->idx =     a->mean() + sqrt(2.0 / a->T * log(max(max(EULER, log(t+1.0)), pow(log(t+1.0), logpower)  * (t+1.0) / lookup.lookup(a->i)))); 
  a->max_idx = a->mean() + sqrt(2.0 / a->T * log(max(max(EULER, log(n+1.0)), pow(log(n+1.0), logpower)  * (n+1.0) / lookup.lookup(a->i)))); 
}

void AnytimeOCUCB::update(vector<Arm>::iterator a) {
  lookup.update(a->i);
}

void AOCUCB::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + sqrt(2.0 / a->T * log((double)t / a->T));
  a->max_idx = a->mean() + sqrt(2.0 / a->T * log((double)n / a->T));
}

void GaussianTS::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + dist(gen) / sqrt(a->T);
  a->max_idx = std::numeric_limits<double>::max();
}

void GaussianGittins::set_index(vector<Arm>::iterator a, uint64_t t) {
  a->idx = a->mean() + table.get_idx(n - t, a->T);
  a->max_idx = a->idx;
}

void GaussianGittinsApprox::set_index(vector<Arm>::iterator a, uint64_t t) {
  uint64_t m = n - t;
  double beta = max(1.0, min(m / pow(log(m), 1.5) / 4.0, m / 4.0 / a->T / pow(log(m/a->T), 0.5)));
  a->idx = a->mean() + sqrt(2.0 / a->T * log(beta));
  a->max_idx = a->idx;
}




