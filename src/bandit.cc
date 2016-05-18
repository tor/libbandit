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


#include "bandit.h"

#include <cfloat>
#include <iostream>

using namespace std;

void BanditProblem::setup() {
  double max_mu = -DBL_MAX;
  for (int i = 0;i != K;++i) {
    if (mean(i) > max_mu) {
      max_mu = mean(i);
    }
  }
  for (int i = 0;i != K;++i) {
    gaps.push_back(max_mu - mean(i));
  }
  regret = 0.0;
}


double BanditProblem::get_regret()const {
  return regret;
}



double BanditProblem::gap(int i)const {
  return gaps[i];
}

