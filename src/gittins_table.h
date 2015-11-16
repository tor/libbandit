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

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <cassert>

class GittinsTable {
  public:
  GittinsTable(std::string fn) {
    std::ifstream in(fn, std::ios::in|std::ios::binary|std::ios::ate);
    assert(in);
    std::streamsize s = in.tellg();
    in.seekg(0, std::ios::beg);
    data = std::vector<double>(s / sizeof(double));
    in.read((char*)data.data(), s);
    in.close();
    n = std::round(0.5*(sqrt(1 + 8 * data.size()) - 1));
  }

  GittinsTable(uint64_t h) : data(h*(h+1)/2, 0.0), n(h) {
  }

  double get_idx(uint64_t m, uint64_t T)const {
    assert(m >= 1);
    assert(T >= 1);
    assert(m + T <= n + 1);
    uint64_t i = (n - m) * (n - m + 1) / 2 + T - 1;
    return data[i];
  }

  void set_idx(uint64_t m, uint64_t T, double v) {
    assert(m >= 1);
    assert(T >= 1);
    assert(m + T <= n + 1);
    uint64_t i = (n - m) * (n - m + 1) / 2 + T - 1;
    data[i] = v;
  }

  void write(std::string fn)const {
    std::ofstream out(fn, std::ios::out | std::ios::binary);
    assert(out);
    out.write((char*)data.data(), sizeof(double) * data.size());
    out.close();
  }

  private:
  std::vector<double> data;
  uint64_t n;
};




