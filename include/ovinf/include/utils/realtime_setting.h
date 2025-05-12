#ifndef REALTIME_SETTING_HPP
#define REALTIME_SETTING_HPP

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

bool setProcessHighPriority(unsigned int priority);
bool StickThisThreadToCore(int core_id);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // !REALTIME_SETTING_HPP
