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


/************************************************************
This is a useful helper class for many bsndit algorithms
that helps with tracking mean/cumulative rewards and such.
************************************************************/

#pragma once 

#include <cstdint>
#include <cmath>

class Arm {
  public:

  int i;
  double idx;
  double reward;
  uint64_t T;
  
  void pull(double r) {
    ++T;
    reward+=r;
  }
  
  double mean()const {
    return (T == 0)?0.0:(reward / T);
  }

  bool operator<(const Arm &b)const {
    return idx < b.idx;
  }

  bool operator>(const Arm &b)const {
    return idx > b.idx;
  }

  Arm(int i, double idx) {
    this->i = i;
    this->idx = idx;

    T = 0;
    reward = 0.0;
  }
};



