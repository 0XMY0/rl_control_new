#include "utils/realtime_setting.h"

#include <linux/sched.h>
#include <linux/sched/types.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>

extern long syscall(long number, ...);

long sched_setattr(pid_t pid, const struct sched_attr *attr,
                   unsigned int flags) {
  return syscall(__NR_sched_setattr, pid, attr, flags);
}

bool setProcessHighPriority(unsigned int priority) {
  /* Get high priority */
  struct sched_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.sched_policy = SCHED_RR;
  attr.sched_priority = priority;
  if (sched_setattr(0, &attr, 0) < 0) {
    printf("sched_setattr failed\n");
    return false;
  }
  return true;
}
