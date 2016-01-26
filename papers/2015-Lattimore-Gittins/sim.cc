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
void do_experiment1(default_random_engine gen, int n, int K, double m, string fn) {
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/10000.bin");
  int xn = 200;
  double step = m/xn;
  for (int t = 0;t!=20000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (double delta = step;delta <= m+0.0001;delta+=step) {
      vector<double> mus = {0};
      for (int k = 0;k != K-1;++k) {
        mus.push_back(-delta);
      }
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, alg_ucb(bandit, n, 2.0)));
      log.log(LogEntry(1, delta, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(2, delta, alg_gaussian_ts(bandit, n, gen)));
      log.log(LogEntry(3, delta, alg_gaussian_gittins(bandit, n, table)));
    }

    if (t % 200 == 0) {
      log.save();
    }
  }
  log.save();
}

/********************************************************
* FIXED HORIZON
* GAUSSIAN REWARDS
* VARIABLE DELTA
********************************************************/
void do_experiment2(default_random_engine gen, string fn) {
  int n = 1000;
  int K = 5;
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/5000.bin");
  for (int t = 0;t!=10000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (double delta = 0.04;delta < 2.0;delta+=0.04) {
      vector<double> mus = {0};
      for (int k = 0;k != K-1;++k) {
        mus.push_back(-delta);
      }
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, alg_gaussian_gittins(bandit, n, table)));
      log.log(LogEntry(1, delta, alg_gaussian_gittins_approx(bandit, n)));
    }

    if (t % 200 == 0) {
      log.save();
    }
  }
  log.save();
}

/********************************************************
* FIXED HORIZON
* GAUSSIAN REWARDS
* VARIABLE DELTA
********************************************************/
void do_experiment3(default_random_engine gen, string fn) {
  int n = 50000;
  int K = 5;
  Logger<LogEntry> log(fn);
  for (int t = 0;t!=10000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (double delta = 0.0025;delta < 0.2;) {
      vector<double> mus = {0};
      for (int k = 0;k != K-1;++k) {
        mus.push_back(-delta);
      }
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, alg_gaussian_gittins_approx(bandit, n)));
      log.log(LogEntry(1, delta, alg_ucb(bandit, n, 2.0)));
      log.log(LogEntry(2, delta, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(3, delta, alg_gaussian_ts(bandit, n, gen)));

      if (delta < 0.1) {
        delta+=0.0025;
      }else {
        delta+=0.01;
      }
    }

    if (t % 5 == 0) {
      log.save();
    }
  }
  log.save();
}



/********************************************************
* FIXED HORIZON
* GAUSSIAN REWARDS
* VARIABLE DELTA
********************************************************/
void do_experiment4(default_random_engine gen, string fn) {
  int n = 2000;
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/5000.bin");
  BayesTable bayes_table("bayes/2000.bin");
  for (int t = 0;t!=10000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (double delta = 0.04;delta <= 2.0;) {
      vector<double> mus = {0,-delta};
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(1, delta, alg_gaussian_gittins(bandit, n, table)));
      log.log(LogEntry(2, delta, alg_gaussian_bayes(bandit, n, bayes_table)));

      delta+=0.04;
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
    case 1: do_experiment1(gen,1000,2,2.0,"data/exp1.log"); break;
    case 2: do_experiment1(gen,1000,5,2.0,"data/exp2.log"); break;
    case 3: do_experiment1(gen,1000,10,2.0,"data/exp3.log"); break;
    case 4: do_experiment1(gen,10000,2,0.5,"data/exp4.log"); break;
    case 5: do_experiment1(gen,10000,5,0.5,"data/exp5.log"); break;
    case 6: do_experiment1(gen,10000,10,0.5,"data/exp6.log"); break;
    case 7: do_experiment2(gen, "data/exp7.log"); break;
    case 8: do_experiment3(gen, "data/exp8.log"); break;
    case 9: do_experiment4(gen, "data/exp9.log"); break;
  }
}



