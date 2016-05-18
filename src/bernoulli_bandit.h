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


#pragma once 

#include <cstdint>
#include <vector>
#include <random>



class BernoulliBandit : public BanditProblem {
  public:
  
  BernoulliBandit(std::vector<double> means, std::default_random_engine &g) : gen(g) {
    this->K = means.size();
    this->means = means;
    setup();
  }

  double sample(int i) {
    std::bernoulli_distribution dist(means[i]);
    return dist(gen);
  }

  double mean(int i)const {
    return means[i];
  }

  void reset() {
    set_regret(0);
  }

  private:
  std::default_random_engine &gen;

  std::vector<double> means;
};


