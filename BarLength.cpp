#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include "OneNetwork.h"
//#include "TwoNetwork.h"

using namespace std;
using filesystem::directory_iterator;

void read(float numbers[IN], string filepath)
{
    ifstream file;
    float a;
    int cnt = 0;
    file.open(filepath);
    while (file >> a) {
        numbers[cnt++] = a;
    }
    file.close();
}

void readBars(void)
{
    OneNetwork network;
    //TwoNetwork network;
    vector<string> items;
    for (const auto& file : directory_iterator("E:\\ints0\\")) {
        items.push_back(file.path().string());
    }
    
    //auto rng = default_random_engine{};
    //shuffle(begin(items), end(items), rng);

    for (int i = 0; i < 99900; i++) {
        float bar[100];
        read(bar, items[i]);
        int num = stoi(items[i].substr(16, 3));
        float target[1];
        if (num != 0) {
            target[0] = (float)num / 1000.0f;
        }
        else {
            target[0] = 0.0001f;
        }
        network.train(bar, target);
        cout << "trained " << i << endl;
    }
    for (int i = 99900; i < 99910; i++) {
        float bar[100];
        read(bar, items[i]);
        int num = stoi(items[i].substr(16, 3));
        float target[1];
        network.query(bar, target);
        cout << items[i] << " - " << target[0] << endl;
    }
}

int main(void)
{
    readBars();
    return 0;
}
