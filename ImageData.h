#pragma once
#include <string>
#include <vector>

using namespace std;

class ImageData
{
private:
	const string path;
	vector<string> items;
	int nItems;
	int current;
public:
	ImageData(string ipath);
	bool nextItem(int* label, float data[32 * 32]);
};

