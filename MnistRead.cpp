#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MnistReader.h"

using namespace std;
using namespace cv;

int main()
{
    const char* imagePath = "E:\\mnist\\train-images.idx3-ubyte";
    const char* labelPath = "E:\\mnist\\train-labels.idx1-ubyte";

    int label;
    uint8_t image[28 * 28];
    MnistReader* mnistReader;

    try {
        mnistReader = new MnistReader(imagePath, labelPath);
    }
    catch (invalid_argument ex) {
        cout << ex.what() << endl;
        return 0;
    }
    mnistReader->info();

    for (int i = 1; i <= 60000; i++) {
        mnistReader->nextlabel(&label);
        mnistReader->nextimage(image);
        stringstream strm;
		strm << "E:\\digits32\\d" << setw(5) << setfill('0') << i << "_" << label << ".png";
		Mat img(28, 28, CV_8UC1, image);
		Mat bgd(32, 32, CV_8UC1, Scalar(0));
		img.copyTo(bgd(Rect(2, 2, img.cols, img.rows)));
		imwrite(strm.str(), bgd);

    }
	return 0;
}
