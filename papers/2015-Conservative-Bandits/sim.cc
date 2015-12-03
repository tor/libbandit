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

#include "data.h"
#include "log.h"
#include "algs.h"
#include "gaussian_bandit.h"

#include <cstring>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono;



int run_time;

bool done() {
  static auto start = system_clock::now();
  auto diff = duration_cast<seconds>(system_clock::now() - start);
  cout << "running for " << diff.count() << " seconds\n";
  if (diff.count() >= run_time) {
    return true;
  }
  return false;
}

string time_string() {
  static auto start = system_clock::now();
  auto diff = duration_cast<seconds>(system_clock::now() - start);
  return to_string(diff.count());
}

/********************************************************
* FIXED HORIZON
* GAUSSIAN REWARDS
* VARIABLE DELTA
********************************************************/
void do_experiment1(default_random_engine gen) {
  Logger<LogEntry> log("data/experiment1.log");
  for (int t = 0;t!=20000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (int n = 100; n!= 10000;n+=100) {
      vector<double> mus = {0, 0.1, -0.1, -0.1, -0.1};
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, n, alg_ucb(bandit, n, 2.0)));
      log.log(LogEntry(1, n, alg_conservative_ucb(bandit, n, 0.9, 1.0 / n))); 
    }

    if (t % 20 == 0) {
      log.save();
    }
  }
  log.save();
}





int main(int argc, char *argv[]) {
  /* seed random number generate */
  default_random_engine gen;
  random_device rd;
  gen.seed(rd());

  /* check that we have three arguments */
  if (argc !=3 ) {
    cout << "bad parameters\n";
    return 0;
  }

  int exp_id = atoi(argv[1]);

  run_time = atoi(argv[2]);

  switch (exp_id) {
    case 1: do_experiment1(gen); break;
  }
}



