#include <string>	
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "ImageData.h"

using namespace std;
using namespace cv;
using filesystem::directory_iterator;

ImageData::ImageData(string ipath) : path(ipath)
{
    for (const auto& file : directory_iterator(path)) {
        items.push_back(file.path().string());
    }
    //auto rng = default_random_engine{};
    //shuffle(items.begin(), items.end(), rng);
    nItems = items.size();
    current = 0;
}

bool ImageData::nextItem(int* label, float data[32 * 32])
{
    if (current >= nItems) {
        return false;
    }
    *label = stoi(items[current].substr(19, 1));
    Mat mat = imread(items[current], IMREAD_GRAYSCALE);
    for (int i = 0; i < (32 * 32); i++) {
        data[i] = (float)((int)mat.data[i] / 255.0f);
    }
    current++;
    return true;
}

