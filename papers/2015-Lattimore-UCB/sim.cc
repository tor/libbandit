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
  int xn = 100;
  double step = m/xn;
  for (int t = 0;t!=200000 && !done();++t) {
    for (double delta = step;delta <= m+0.0001;delta+=step) {
      vector<double> mus = {0};
      for (int k = 0;k != K-1;++k) {
        mus.push_back(-delta);
      }
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
//      log.log(LogEntry(0, delta, alg_ucb(bandit, n, 2.0)));
//      log.log(LogEntry(1, delta, alg_ocucb(bandit, n, 3.0, 2.0)));
//      log.log(LogEntry(2, delta, alg_aocucb(bandit, n, 2.0)));
 //     log.log(LogEntry(3, delta, alg_moss(bandit, n, 2.0)));
      log.log(LogEntry(5, delta, alg_gaussian_gittins(bandit, n, table)));
//      log.log(LogEntry(6, delta, alg_gaussian_ts(bandit, n, gen)));
    }
    log.save(false);
  }
  log.save();
}


/********************************************************
* ANALYSE ALPHA
********************************************************/
void do_experiment2(default_random_engine &gen, int n, int K, string fn) {
  Logger<LogEntry> log(fn);
  double m = 2.0;
  int xn = 200;
  double step = m/xn;
  for (int t = 0;t!=20000 && !done();++t) {
    for (double delta = step;delta <= m+0.0001;delta+=step) {
      vector<double> mus = {0};
      for (int i = 0;i != K-1;++i) {
        mus.push_back(-delta);
      }
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, alg_ocucb(bandit, n, 1.0, 2.0)));
      log.log(LogEntry(1, delta, alg_ocucb(bandit, n, 2.0, 2.0)));
      log.log(LogEntry(2, delta, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(3, delta, alg_ocucb(bandit, n, 6.0, 2.0)));
    }

    log.save(false);
  }
  log.save();
}


/********************************************************
* ANALYSE ALPHA
********************************************************/
void do_experiment3(default_random_engine &gen, int K, string fn) {
  Logger<LogEntry> log(fn);
  for (int t = 0;t!=20000 && !done();++t) {
    for (uint64_t n = 5000;n <= 100000;n+=5000) {
      vector<double> mus = {0};
      for (int i = 0;i != K-1;++i) {
        mus.push_back(-0.2);
      }
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, n, alg_ocucb(bandit, n, 1, 2.0)));
      log.log(LogEntry(1, n, alg_ocucb(bandit, n, 1.5, 2.0)));
      log.log(LogEntry(2, n, alg_ocucb(bandit, n, 2.0, 2.0)));
      log.log(LogEntry(3, n, alg_ocucb(bandit, n, 2.5, 2.0)));
      log.log(LogEntry(4, n, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(5, n, alg_ocucb(bandit, n, 4.0, 2.0)));
      log.log(LogEntry(6, n, alg_ocucb(bandit, n, 6.0, 2.0)));
      log.log(LogEntry(7, n, alg_ocucb(bandit, n, 8.0, 2.0)));
    }

    log.save(false);
  }
  log.save();
}



/********************************************************
* FIXED DELTA
* GAUSSIAN REWARDS
* VARIABLE HORIZON
********************************************************/
void do_experiment4(default_random_engine gen, int K, string fn) {
  Logger<LogEntry> log(fn);
  double delta=0.3;
  vector<double> mus = {0};
  for (int k = 0;k != K-1;++k) {
    mus.push_back(-delta);
  }
  for (int t = 0;t!=20000 && !done();++t) {
    for (int n = 100;n<=100000;) {
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
/*      log.log(LogEntry(0, n, alg_ucb(bandit, n, 2.0)));
      log.log(LogEntry(1, n, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(2, n, alg_aocucb(bandit, n, 2.0)));
      log.log(LogEntry(3, n, alg_moss(bandit, n, 2.0)));*/
      log.log(LogEntry(5, n, alg_gaussian_ts(bandit, n, gen)));
      if (n < 1000) {
        n+=100;
      }else if (n < 10000) {
        n+=1000;
      }else {
        n+=10000;
      }
    }
    log.save(false);
  }
  log.save();
}


/********************************************************
* MOSS FAILURE
* GAUSSIAN REWARDS
* VARIABLE HORIZON
********************************************************/
void do_experiment5(default_random_engine gen, string fn) {
  Logger<LogEntry> log(fn);
  for (uint64_t K : {5, 10, 50, 100, 150, 200, 400, 600, 800, 1000, 1500}) {
    double delta = 0.25 / (double)K;
    vector<double> mus = {0,-delta};
    for (int k = 0;k != K-2;++k) {
      mus.push_back(-1.0);
    }

    uint64_t n = K * K * K;

    shuffle(mus.begin(),mus.end(), gen);
    GaussianBandit bandit(mus, gen);

    double r_ocucb = alg_ocucb(bandit, n, 3.0, 2.0);
    double r_moss = alg_moss(bandit, n, 4.0);
      
    cout << K << " ocucb=" << r_ocucb << " moss=" << r_moss << "\n";

    log.log(LogEntry(0, K, r_ocucb));
    log.log(LogEntry(1, K, r_moss));

    log.save(false);
  }
  log.save();
}


/********************************************************
* VARIABLE DELTA
* GAUSSIAN REWARDS
* VARIABLE HORIZON
********************************************************/
void do_experiment6(default_random_engine gen, int K, string fn) {
  Logger<LogEntry> log(fn);
  double delta=1.0 / K;
  vector<double> mus = {0};
  for (int k = 0;k != K-1;++k) {
    mus.push_back(-delta * (k+1));
  }
  for (int t = 0;t!=20000 && !done();++t) {
    for (int n = 100;n<=100000;) {
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, n, alg_ucb(bandit, n, 2.0)));
      log.log(LogEntry(1, n, alg_ocucb(bandit, n, 3.0, 2.0)));
      log.log(LogEntry(2, n, alg_aocucb(bandit, n, 2.0)));
      log.log(LogEntry(3, n, alg_moss(bandit, n, 2.0)));
      log.log(LogEntry(5, n, alg_gaussian_ts(bandit, n, gen)));
      if (n < 1000) {
        n+=100;
      }else if (n < 10000) {
        n+=1000;
      }else {
        n+=10000;
      }
    }
    log.save(false);
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
    case 1: do_experiment1(gen,10000,   2,   1.0,  "data/exp1.log"); break;
    case 2: do_experiment1(gen,10000,   10,  1.0,  "data/exp2.log"); break;
    case 3: do_experiment1(gen,10000,   100, 1.0,  "data/exp3.log"); break;
    case 4: do_experiment2(gen,10000,    2,        "data/exp4.log"); break;
    case 5: do_experiment2(gen,10000,    10,       "data/exp5.log"); break;
    case 6: do_experiment3(gen,2,                  "data/exp6.log"); break;
    case 7: do_experiment3(gen,10,                 "data/exp7.log"); break;
    case 8: do_experiment4(gen,2,                  "data/exp8.log"); break;
    case 9: do_experiment4(gen,10,                 "data/exp9.log"); break;
    case 10: do_experiment5(gen,                   "data/exp10.log"); break;
    case 11: do_experiment6(gen,10,                "data/exp11.log"); break;
    case 12: do_experiment6(gen,100,               "data/exp12.log"); break;
  }
}



