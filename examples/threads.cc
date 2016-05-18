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
Compute mean regret of UCB using threads
*************************************************/

/* load some algorithms */
#include "algs.h"      

/* load the bandit class */
#include "gaussian_bandit.h"

/* load a simple thread pool */
#include "pool.h"

#include <vector>
#include <iostream>
#include <random>

using namespace std;

int main() {
  int max_threads = 8;


  /* we want 100 samples */
  int samples = 100;

  /* horizon of 5000000 */
  int horizon = 5000000;

  /* two arms, one with mean 0, the other with mean 0.1 */
  vector<double> means = {0, 0.1};            


  /* create job list with `max_threads` threads at most, returning doubles */
  Pool<double> pool(max_threads);
  
  for (int i = 0;i != samples;++i) {
    pool.push([means,horizon] {
      /* for seeding our random number generator */
      random_device rd;

      /* we need random numbers */
      default_random_engine gen(rd());                  



      /* create a gaussian bandit */
      GaussianBandit bandit(means, gen);
      UCB ucb(2.0);

      /* create a UCB algorithm with alpha = 2 */
      return ucb.sim(bandit, horizon);
    });
  }
  
  vector<double> data = pool.run();

  double R_ucb = 0.0;

  for (auto r : data) {
    R_ucb+=r;
  }

  /* output the average regret */
  cout << "average regret of UCB is " << R_ucb / samples << "\n";

  return 0;
}






