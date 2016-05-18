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

#include "bandit.h"
#include "gittins_table.h"
#include "arm.h"

#include <cstdint>
#include <random>
#include <vector>


/*************************************************************
* DATA STRUCTURE FOR UPDATING B_i(t) in OCUCB-n
*************************************************************/
class SortedLookup {
  public:
  SortedLookup() {
  }
  SortedLookup(int s, double rho) {
    x.push_back(0);
    L.push_back(0.0);
    next.push_back(1);
    last.push_back(-1);
    for (int i = 0;i != s;++i) {
      x.push_back(1);
      next.push_back(i+2);
      last.push_back(i);
      L.push_back(s);
    }
    next.push_back(-1);
    last.push_back(s);
    L.push_back(0.0);
    x.push_back(std::numeric_limits<uint64_t>::max());
    this->rho = rho;
  }

  void update(int i) {
    i++;
    x[i]++;
    int n = next[i];
    while (x[i] > x[n]) {
      n = next[n]; 
    }
    last[next[i]] = last[i];
    next[last[i]] = next[i];

    next[i] = n;
    last[i] = last[n];

    next[last[n]] = i;
    last[n] = i;
    while (n != -1) {
      L[i]+= pow((double)x[i], rho) - pow((double)x[i] - 1.0, rho);
      L[n]+= pow((double)x[i], rho) - pow((double)x[i] - 1.0, rho);
      n = next[n];
    }
  }

  double lookup(int i) {
    return pow((double)x[i+1], 1.0 - rho) * L[i+1];
  }

  std::vector<uint64_t> x;
  std::vector<int> next;
  std::vector<int> last;
  std::vector<double> L;
  double rho;
};


class IndexAlgorithm {
  public:
  double sim(BanditProblem &bp, uint64_t horizon);

  protected:
  virtual void set_index(std::vector<Arm>::iterator, uint64_t t) = 0;

  virtual void update(std::vector<Arm>::iterator) {
  }

  uint64_t n;
  uint64_t K;

  std::vector<Arm> arms;

  private:
};


class UCB : public IndexAlgorithm {
  void set_index(std::vector<Arm>::iterator, uint64_t t);
};

class MOSS : public IndexAlgorithm {
  void set_index(std::vector<Arm>::iterator, uint64_t t);
};

class OCUCB : public IndexAlgorithm {
  void set_index(std::vector<Arm>::iterator, uint64_t t);
};

class AOCUCB : public IndexAlgorithm {
  void set_index(std::vector<Arm>::iterator, uint64_t t);
};

class AnytimeOCUCB : public IndexAlgorithm {
  public:
  AnytimeOCUCB(double rho, double logpower) : rho(rho), logpower(logpower) {
  }

  double sim(BanditProblem &bp, uint64_t horizon) {
    lookup = SortedLookup(bp.K, rho);
    return IndexAlgorithm::sim(bp, horizon);
  }
  protected:
  void update(std::vector<Arm>::iterator); 
  void set_index(std::vector<Arm>::iterator, uint64_t t);

  double rho;
  double logpower;

  SortedLookup lookup; 
};

class GaussianTS : public IndexAlgorithm {
  public:
  GaussianTS(std::default_random_engine &gen) : gen(gen), dist(0.0, 1.0) {
  }
  private:
  void set_index(std::vector<Arm>::iterator, uint64_t t);
  std::default_random_engine &gen;
  std::normal_distribution<double> dist;
};

class GaussianGittins : public IndexAlgorithm {
  public:
  GaussianGittins(std::string fn) : table(fn) {
  }
  private:
  void set_index(std::vector<Arm>::iterator, uint64_t t);
  GittinsTable table;
};

class GaussianGittinsApprox : public IndexAlgorithm {
  void set_index(std::vector<Arm>::iterator, uint64_t t);
};












