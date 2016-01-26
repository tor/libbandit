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

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <string>
#include <cerrno>
#include <random>
/* yuck, no C++ versions? */
#include <fcntl.h>
#include <unistd.h>
#include <chrono>


class LogEntry {
  public:

  int id;
  double x;
  double y;
  LogEntry(int idc, double xc, double yc) : id(idc), x(xc), y(yc) {
  }
  LogEntry() {
  }
};

template<class T> class Logger {
  public:
  std::vector<T> data;
  std::string filename;

  int sleep_time;
  std::chrono::time_point<std::chrono::system_clock> last_save;
  std::default_random_engine gen;

  Logger(std::string fn) {
    std::random_device rd;
    gen.seed(rd());
    reset_clock();
    filename = fn;
  }


  void reset_clock() {
    std::uniform_int_distribution<int> dist(10, 20);
    last_save = std::chrono::system_clock::now();
    sleep_time = dist(gen);
    std::cout << "saving in " << sleep_time << " second\n";
  }

  int time_since_save() {
    auto diff = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - last_save);
    return diff.count();
  }

  void give_lock() {
    remove(".lock");
  }

  bool get_lock(bool wait) {
    int fd = open(".lock", O_WRONLY | O_CREAT | O_EXCL, S_IRWXU);
    while (fd < 0) {
      std::cout << "I found a lock file\n";
      reset_clock();
      if (!wait) {
        return false;
      }
      sleep(1);
      std::cout << "sleeping on lock\n";
      fd = open(".lock", O_WRONLY | O_CREAT | O_EXCL, S_IRWXU);
    }
    close(fd);
    return true;
  }

  void log(T entry) {
    data.push_back(entry);
  }

  void save(bool force = true) {
    if (!force && time_since_save() < sleep_time) {
      return;
    }
    if (!get_lock(force)) {
      return;
    }
    std::cout << "saving to " << filename << "\n";
    std::ofstream out(filename, std::ios::app);
    for (auto &e : data) {
      out.write((char*)&e, sizeof(T));
    }
    out.close();
    give_lock();
    data.clear();
    reset_clock();
  }

};


