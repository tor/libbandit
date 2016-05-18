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


/******************************************************
* This is designed to read the binary data files and 
* output a plain text table suitable for plotting.
******************************************************/
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <set>
#include <map>
#include <cstdint>

#include "data.h"
#include "log.h"

using namespace std;




int main(int argc, char* argv[]) {
  if (argc != 2) {
    cout << "bad arguments\n";
    return 0;
  }
  map<double,map<int,Data> > table;
  string line;
  ifstream in(argv[1], ios::in | ios::ate);
  auto end = in.tellg();
  uint64_t size = end;
  uint64_t pos = 0;
  uint64_t block_size = 500000;
  vector<LogEntry> data(block_size);
  in.seekg(0, ios::beg);
  while (pos != size) {
    uint64_t read_number = block_size;
    if ((size - pos) / sizeof(LogEntry) < block_size) {
      read_number = (size - pos) / sizeof(LogEntry);
    }
    cerr << "reading " << read_number << "\n";
    in.read((char*)data.data(), sizeof(LogEntry) * read_number);
    pos+=sizeof(LogEntry) * read_number;


    for (uint64_t i = 0;i != read_number;++i) {
      auto iter1 = table.find(data[i].x);
      if (iter1 == table.end()) {
        table[data[i].x] = {{data[i].id, Data(data[i].y)}};
      }else {
        auto iter2 = iter1->second.find(data[i].id);
        if (iter2 == iter1->second.end()) {
          iter1->second[data[i].id] = Data(data[i].y);
        }else {
          iter2->second << data[i].y;
        }
      }
    }
  }
  cout << "\%x ";
  for (auto &alg : table.begin()->second) {
    cout << alg.first << " ";
  }
  cout << "\n";
  for (auto &x : table) {
    cout << x.first << " ";
    for (auto &alg : x.second) {
      cout << alg.second.mean() << " ";
    }
    for (auto &alg : x.second) {
      cout << 2.0 * alg.second.standard_error() << " ";
    }
    for (auto &alg : x.second) {
      cout << alg.second.size() << " ";
    }
    for (auto &alg : x.second) {
      cout << alg.second.quantile(0.9) << " ";
    }
    cout << "\n";
  }
}










