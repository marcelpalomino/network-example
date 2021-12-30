#include <iostream>
#include <filesystem>  
#include "ImageData.h"
//#include "OneNetwork.h"
#include "TwoNetwork.h"

using namespace std;

int main(void)
{
    string path = "E:\\digits32\\";
    ImageData imageData(path);
    //OneNetwork network;
    TwoNetwork network;

    for (int i = 0; i < 4000; i++) {
        int label;
        float data[1024];
        float target[10] = { .01f, .01f, .01f, .01f, .01f, .01f, .01f, .01f, .01f, .01f };
        if (imageData.nextItem(&label, data)) {
            target[label] = 1.0f;
            network.train(data, target);
        }
    }
    for (int i = 0; i < 100; i++) {
        int label;
        float data[1024];
        float target[10];
        if (imageData.nextItem(&label, data)) {
            network.query(data, target);
            for (int i = 0; i < 10; i++) {
                cout << target[i] << " ";
            }
            cout << endl;
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    cout << setw(2) << hex << (int)(data[i * 32 + j] * 255);
                }
                cout << endl;
            }
        }
    }
    return 0;
}