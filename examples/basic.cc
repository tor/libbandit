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


/*************************************************
A simple demo example comparing UCB and OCUCB
*************************************************/

#include "gaussian_bandit.h"
#include "bernoulli_bandit.h"
#include "algs.h"
#include "gittins_table.h"

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

int main() {
  random_device rd;
  default_random_engine gen(rd());
  uint64_t K = 2;
  uint64_t n = 10000;
  vector<double> mus = {0};
  for (int i = 1;i != K;++i) {
    mus.push_back(-0.3);
  }
  

  GaussianBandit bandit(mus, gen); 
  vector<double> r(5, 0.0);
  UCB ucb;
  OCUCB ocucb;
  AnytimeOCUCB aocucb(0.5, 0.0);
  GaussianTS ts(gen);
  GaussianGittins git("gittins/10000.bin");
  for (int samples = 1;samples <= 1000;++samples) {
    r[0]+=ucb.sim(bandit, n);
    r[1]+=ocucb.sim(bandit, n);
    r[2]+=aocucb.sim(bandit, n);
    r[3]+=ts.sim(bandit, n);
    r[4]+=git.sim(bandit, n);
    if (samples % 100 == 0) {
      for (int i = 0;i != 5;++i) {
        cout << r[i] / (samples+1) << " ";
      }
      cout << "\n";
    }
  }
  return 0;
}






