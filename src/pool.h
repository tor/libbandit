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



/********************************************************************
A simple thread pool. Useful for running simulations on multi-core
CPUs.

Basic usage:

Pool<returntype> pool(max_threads);

Call pool.add() to add jobs to the queue

Call pool.run(), which runs jobs and returns a vector<returntype> of
the results.
********************************************************************/
#pragma once


#include <queue>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <functional>
#include <chrono>
#include <future>

template<class T> class Pool {
  public:
  unsigned int max_threads;

  Pool(unsigned int max_threads) {
    this->max_threads = max_threads;
  }

  void push(std::function<T()> job) {
    jobs.push(job);
  }

  std::vector<T> run();

  std::queue<std::function<T()>> jobs;
};

template<class T>
std::vector<T> Pool<T>::run() { 
  std::vector<std::future<T>> threads;
  std::vector<T> data;

  std::cout << "running pool with " << jobs.size() << " jobs\n";

  int running = 0;
  
  while (jobs.size() > 0) {
    if (threads.size() < max_threads) {
      running++;
      auto job = jobs.front();
      std::cout << "starting job " << running << "\n";
      jobs.pop();
      threads.push_back(std::async(std::launch::async, [job] {
        return job();
      }));
    }

    for (auto f = threads.begin();f!=threads.end();) {
      auto status = f->wait_for(std::chrono::milliseconds(0));
      if (status == std::future_status::ready) {
        data.push_back(f->get());
        f = threads.erase(f);
      }else {
        ++f;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  }

  for (auto &f : threads) {
    data.push_back(f.get());
  }

  return data;
}



