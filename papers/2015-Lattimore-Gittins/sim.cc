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
void do_experiment1(default_random_engine gen, int K) {
  int n = 1000;
  string fn;
  switch (K) {
    case 2: fn = "data/experiment1.log";break;
    case 5: fn = "data/experiment2.log";break;
    case 10: fn = "data/experiment3.log";break;
  }
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/5000.bin");
  for (int t = 0;t!=20000 && !done();++t) {
    cout << "running trial: " << t << "\n";
    for (double delta = 0.04;delta < 2.0;delta+=0.04) {
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
void do_experiment4(default_random_engine gen) {
  int n = 1000;
  int K = 5;
  string fn = "data/experiment4.log";
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
void do_experiment5(default_random_engine gen) {
  int n = 50000;
  int K = 5;
  string fn = "data/experiment5.log";
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/5000.bin");
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
void do_experiment6(default_random_engine gen) {
  int n = 400;
  string fn = "data/experiment6.log";
  Logger<LogEntry> log(fn);
  GittinsTable table("gittins/5000.bin");
  BayesTable bayes_table("gittins/bayes400.bin");
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

    if (t % 5 == 0) {
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
    case 1: do_experiment1(gen,2); break;
    case 2: do_experiment1(gen,5); break;
    case 3: do_experiment1(gen,10); break;
    case 4: do_experiment4(gen); break;
    case 5: do_experiment5(gen); break;
    case 6: do_experiment6(gen); break;
  }
}



