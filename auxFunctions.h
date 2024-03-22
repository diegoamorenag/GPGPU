
#ifndef AUX_FUNC
#define AUX_FUNC

#include <functional>
#include <chrono>
#include <iostream>

std::chrono::duration<double>::rep Time(std::function<void()> func);

#endif // AUX_FUNC