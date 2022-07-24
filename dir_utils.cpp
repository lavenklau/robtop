#include "dir_utils.h"
#include "regex"



std::vector<std::string> dir_utils::filterFiles(const std::vector<std::string>& files, std::string extname)
{
	std::vector<std::string> filteredfiles;
	for (int i = 0; i < files.size(); i++) {
		std::filesystem::path pth(files[i]);
		if (pth.extension().string() == extname) {
			filteredfiles.emplace_back(pth.string());
		}
	}
	return filteredfiles;
}

std::vector<std::string> dir_utils::matchFiles(const std::vector<std::string>& files, std::string regPattern)
{
	std::vector<std::string> matchedfiles;
	std::regex ptn(regPattern);
	for (int i = 0; i < files.size(); i++) {
		std::filesystem::path pth(files[i]);
		std::string fn = pth.filename().string();
		if (std::regex_match(fn, ptn)) {
			matchedfiles.emplace_back(files[i]);
		}
	}
	return matchedfiles;
}

std::vector<std::string> dir_utils::listFile(std::string dirname)
{
	std::filesystem::path pth(dirname);
	std::filesystem::directory_iterator contents(pth);

	std::vector<std::string> files;

	for (auto it : contents) {
		if (it.is_directory()) continue;
		
		files.emplace_back(it.path().string());
	}
	
	return files;
}

std::string dir_utils::path2filename(std::string pathstr)
{
	std::filesystem::path p(pathstr);
	return p.filename().string();
}

std::string dir_utils::path2extension(std::string pathstr)
{
	std::filesystem::path p(pathstr);
	return p.extension().string();
}
