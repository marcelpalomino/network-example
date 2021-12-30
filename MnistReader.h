#pragma once
#include <string>
using namespace std;

class MnistReader
{
private:
	const char* imagepath;
	const char* labelpath;
	FILE* imagefile;
	FILE* labelfile;
	int imagic, lmagic, images, labels, rows, cols;
public:
	MnistReader(const char* ipath, const char* lpath);
	~MnistReader();
	void info(void);
	void nextimage(uint8_t image[28*28]);
	void nextlabel(int* label);
};