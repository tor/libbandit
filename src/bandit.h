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
#include <cstdint>

/*********************************************************
* inherit from this class if you want to define a new 
* noise model. Must implement access to the mean for
* the purpose of computing the regret. Must also
* implement a way to sample from the ith arm.
*********************************************************/
class BanditProblem {
  public:
  /* returns a sample from the ith arm */
  virtual double        sample(int i) = 0;         
  /* returns the mean of the ith arm */
  virtual double        mean(int i)const = 0;           
  /* resets */
  virtual void reset() = 0;

  /* returns the gap for the ith arm */
  double gap(int i)const;                        

  /* returns the regret */
  double get_regret()const;

  /* initialises the gaps */
  void setup();                              

  void set_regret(double r) {
    regret = r;
  }

  /* choose arm */
  double choose(int i) {
    regret+=gap(i);
    return sample(i);
  }

  int K;

  private:
  std::vector<double> gaps;
  double regret;
};



