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

#include <cstdint>
#include <random>
#include <vector>

double alg_ocucb(BanditProblem &bp, uint64_t n, double alpha, double psi);
double alg_ucb(BanditProblem &bp, uint64_t n, double alpha);
double alg_moss(BanditProblem &bp, uint64_t n, double alpha);
double alg_aocucb(BanditProblem &bp, uint64_t n, double alpha);
double alg_gaussian_ts(BanditProblem &bp, uint64_t n, std::default_random_engine &gen);
double alg_gaussian_gittins(BanditProblem &bp, uint64_t n, GittinsTable &table);
double alg_gaussian_gittins_approx(BanditProblem &bp, uint64_t n); 
double alg_gaussian_bayes(BanditProblem &bp, uint64_t n, BayesTable &table);
double alg_conservative_ucb(BanditProblem &bp, uint64_t n, double alpha, double delta, bool mu_known = true);
double alg_budget_first(BanditProblem &bp, uint64_t n, double alpha, double delta);
double alg_unbalanced_moss(BanditProblem &bp, uint64_t n, std::vector<double> B);

