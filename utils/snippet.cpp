#include "snippet.h"
#include <chrono>
#include <thread>

void snippet::trim(std::string& str)
{
	str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {return ch != ' '; }));
	str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {return ch != ' '; }).base(), str.end());
}

void snippet::stop_ms(int millisecond)
{
	//Sleep(millisecond);
	std::this_thread::sleep_for(std::chrono::milliseconds(millisecond));
}
