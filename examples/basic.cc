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
#include "algs.h"
#include "gittins_table.h"

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

int main() {
  /* for seeding our random number generator */
  random_device rd;

  /* we need random numbers */
  default_random_engine gen(rd());                  

  /* horizon of 500 */
  int horizon = 500;

  /* two arms, one with mean 0, the other with mean 0.1 */
  vector<double> means = {0, 0.3};            

  /* load gittins indices */
  GittinsTable gittins("gittins/500.bin");

  /* we want 1000 samples */
  int samples = 1000;

  /* compute average regret over `sample` i.i.d. samples */
  double R_ucb = 0.0;
  double R_ocucb = 0.0;
  double R_aocucb = 0.0;
  double R_ts = 0.0;
  double R_git = 0.0;
  double R_moss = 0.0;
  for (int i = 0;i != samples;++i) {
    shuffle(means.begin(), means.end(), gen);
    /* create a gaussian bandit */
    GaussianBandit bandit(means, gen);

    R_ucb+=alg_ucb(bandit, horizon, 2.0);
    R_ocucb+=alg_ocucb(bandit, horizon, 3.0, 2.0);
    R_aocucb+=alg_aocucb(bandit, horizon, 2.0);
    R_ts+=alg_gaussian_ts(bandit, horizon, gen);
    R_moss+=alg_moss(bandit, horizon, 2.0);
    R_git+=alg_gaussian_gittins(bandit, horizon, gittins);
  }

  /* output the average regret */
  cout << "average regret of UCB is " << R_ucb / samples << "\n";
  cout << "average regret of OCUCB is " << R_ocucb / samples << "\n";
  cout << "average regret of AOCUCB is " << R_aocucb / samples << "\n";
  cout << "average regret of TS is " << R_ts / samples << "\n";
  cout << "average regret of MOSS is " << R_moss / samples << "\n";
  cout << "average regret of Gittins is " << R_git / samples << "\n";

  return 0;
}






