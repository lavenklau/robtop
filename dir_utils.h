#pragma once

#include "vector"
#include "string"
#include "filesystem"

namespace dir_utils {
	std::vector<std::string> filterFiles(const std::vector<std::string>& files, std::string extname);

	std::vector<std::string> matchFiles(const std::vector<std::string>& files, std::string regPattern);

	std::vector<std::string> listFile(std::string dirname);

	std::string path2filename(std::string pathstr);

	std::string path2extension(std::string pathstr);
};
