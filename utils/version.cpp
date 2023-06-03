#ifdef USE_VERSION

#include "version.h"

#else

#define GIT_VERSION_HASH

#endif

#pragma push_macro("TMP")
#pragma push_macro("STR")

#define TMP(m) #m
#define STR(v) TMP(v)

#include <iostream>
#include <string>

void version_info(void) {
	std::cout << "[Version] :" << " " STR(GIT_VERSION_HASH) "" << std::endl;
}

std::string version_hash(void) {
	return " " STR(GIT_VERSION_HASH)"";
}

#pragma pop_macro("STR")
#pragma pop_macro("TMP")
