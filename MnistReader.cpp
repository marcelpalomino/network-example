#include <stdexcept>
#include <iostream>
#include "MnistReader.h"
using namespace std;

MnistReader::MnistReader(const char* ipath, const char* lpath) : imagepath(ipath), labelpath(lpath)
{
	unsigned char cmagic[4], cimages[4], crows[4], ccols[4];
	imagefile = fopen(imagepath, "rb");
	labelfile = fopen(labelpath, "rb");
	if (imagefile == NULL) throw invalid_argument("wrong image path");
	if (labelfile == NULL) throw invalid_argument("wrong label path");
	
	fread(cmagic, 4, 1, imagefile);
	fread(cimages, 4, 1, imagefile);
	fread(crows, 4, 1, imagefile);
	fread(ccols, 4, 1, imagefile);
	imagic = (cmagic[0] << 24) | (cmagic[1] << 16) | (cmagic[2] << 8) | (cmagic[3]);
	images = (cimages[0] << 24) | (cimages[1] << 16) | (cimages[2] << 8) | (cimages[3]);
	rows = (crows[0] << 24) | (crows[1] << 16) | (crows[2] << 8) | (crows[3]);
	cols = (ccols[0] << 24) | (ccols[1] << 16) | (ccols[2] << 8) | (ccols[3]);

	fread(cmagic, 4, 1, labelfile);
	fread(cimages, 4, 1, labelfile);
	lmagic = (cmagic[0] << 24) | (cmagic[1] << 16) | (cmagic[2] << 8) | (cmagic[3]);
	labels = (cimages[0] << 24) | (cimages[1] << 16) | (cimages[2] << 8) | (cimages[3]);
}

MnistReader::~MnistReader()
{
	if (imagefile != NULL) fclose(imagefile);
	if (labelfile != NULL) fclose(labelfile);
}

void MnistReader::info(void)
{
	cout << "imagic: " << imagic << ", images: " << images << ", rows: " << rows << ", cols: " << cols << endl;
	cout << "lmagic: " << lmagic << ", labels: " << labels << endl;
}

void MnistReader::nextimage(uint8_t image[28*28])
{
	unsigned char c;
	for (int i = 0; i < (28 * 28); i++) {
		fread(&c, 1, 1, imagefile);
		image[i] = (uint8_t)c;
	}
}

void MnistReader::nextlabel(int* label)
{
	unsigned char ch;
	fread(&ch, 1, 1, labelfile);
	*label = (int)ch;
}
