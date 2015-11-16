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
/* yuck, no C++ versions? */
#include <fcntl.h>
#include <unistd.h>


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

  Logger(std::string fn) {
    filename = fn;
  }

  void give_lock() {
    remove(".lock");
  }

  void get_lock() {
    int fd = open(".lock", O_WRONLY | O_CREAT | O_EXCL, S_IRWXU);
    while (fd < 0) {
      perror("something went wrong:");
      sleep(1);
      std::cout << "sleeping on lock\n";
      fd = open(".lock", O_WRONLY | O_CREAT | O_EXCL, S_IRWXU);
    }
    close(fd);
  }

  void log(T entry) {
    data.push_back(entry);
  }

  void save() {
    get_lock();
    std::ofstream out(filename, std::ios::app);
    for (auto &e : data) {
      out.write((char*)&e, sizeof(T));
    }
    out.close();
    give_lock();
    data.clear();
  }

};


