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

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

class Data {
  public:

  double mean() {
    double sum = 0.0;
    for (auto x : data) {
      sum+=x / data.size();
    }
    return sum;
  }
  double variance() {
    double m = mean();
    double sum = 0.0;
    for (auto x : data) {
      sum+=pow(x - m, 2.0) / data.size();
    }
    return sum;
  }
  double standard_error() {
    return sqrt(variance() / data.size());
  }

  uint64_t size() {
    return data.size();
  }

  double quantile(double p) {
    if (!sorted) {
      std::sort(data.begin(), data.end());
    }
    sorted = true;
    return data[(int)(p * data.size())];
  }


  std::vector<double> data;

  Data() {
  }

  Data(double x) {
    data.push_back(x);
  }

  friend Data &operator<<(Data &d, double x) {
    d.data.push_back(x);
    d.sorted = false;
    return d;
  }

  bool sorted;
  private:
};


