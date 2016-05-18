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
WORST CASE
********************************************************/
void do_experiment1(default_random_engine gen, int n, int K, double m, string fn) {
  Logger<LogEntry> log(fn);
  int xn = 100;
  double step = m/xn;

  AnytimeOCUCB anytime_ocucb(2.0, 0.5);
  OptAnytimeOCUCB opt_anytime_ocucb(2.0, 0.5); 
  UCB ucb(2.0);
  OCUCB ocucb(3.0, 2.0);
  GaussianGittins gittins("gittins/10000.bin");
  GaussianTS ts(gen);
  
  for (int t = 0;t!=200000 && !done();++t) {
    for (double delta = step;delta <= m+0.0001;delta+=step) {
      vector<double> mus = {0};
      for (int k = 0;k != K-1;++k) {
        mus.push_back(-delta);
      }
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, delta, anytime_ocucb.sim(bandit, n)));
      log.log(LogEntry(1, delta, opt_anytime_ocucb.sim(bandit, n)));
      log.log(LogEntry(2, delta, ucb.sim(bandit, n)));
      log.log(LogEntry(3, delta, ocucb.sim(bandit, n)));
      log.log(LogEntry(4, delta, ts.sim(bandit, n)));
      log.log(LogEntry(5, delta, gittins.sim(bandit, n)));
    }
    log.save(false);
  }
  log.save();
}


/********************************************************
ASYMPTOTIC
********************************************************/
void do_experiment2(default_random_engine gen, string fn) {
  Logger<LogEntry> log(fn);

  AnytimeOCUCB anytime_ocucb(2.0, 0.5);
  OptAnytimeOCUCB opt_anytime_ocucb(2.0, 0.5);
  UCB ucb(2.0);
  OCUCB ocucb(3.0, 2.0);
  GaussianTS ts(gen);

  vector<double> mus = {0,-0.1,-0.1,-0.1,-0.5,-0.5,-0.5,-1.0,-1.0,-1.0};
  
  for (int t = 0;t!=200000 && !done();++t) {
    for (uint64_t n : {1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000}) {
      shuffle(mus.begin(),mus.end(), gen);
      GaussianBandit bandit(mus, gen);
      log.log(LogEntry(0, n, anytime_ocucb.sim(bandit, n)));
      log.log(LogEntry(1, n, opt_anytime_ocucb.sim(bandit, n)));
      log.log(LogEntry(2, n, ucb.sim(bandit, n)));
      log.log(LogEntry(3, n, ocucb.sim(bandit, n)));
      log.log(LogEntry(4, n, ts.sim(bandit, n)));
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
    case 1: do_experiment1(gen, 5000, 2, 0.5, "data/exp1.log"); break;
    case 2: do_experiment1(gen, 5000, 10, 0.75, "data/exp2.log"); break;
    case 3: do_experiment1(gen, 5000, 100, 1.0, "data/exp3.log"); break;
    case 4: do_experiment2(gen, "data/exp4.log"); break;
  }
}



