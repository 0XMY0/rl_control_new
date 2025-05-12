#include "utils/realtime_setting.h"

#ifdef __linux__
#define _GNU_SOURCE
#include <sched.h>
#include <stdbool.h>
// #include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

bool StickThisThreadToCore(int core_id) {
  int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (core_id < 0 || core_id >= num_cores) {
    printf("invalid core_id\n");
    return false;
  }

  size_t size;
  cpu_set_t *cpuset;

  cpuset = CPU_ALLOC(1);
  size = CPU_ALLOC_SIZE(1);

  CPU_ZERO_S(size, cpuset);
  CPU_SET_S(core_id, size, cpuset);

  pthread_t current_thread = pthread_self();
  if (pthread_setaffinity_np(current_thread, size, cpuset) != 0)
    return false;

  return true;
}

#endif /* ifdef __linux__ */
