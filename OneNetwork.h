#pragma once

#define IN 100
#define HI 10
#define OU 1
#define LR 0.2

class OneNetwork
{
private:
	float wInputHidden[HI][IN];
	float wHiddenOutput[OU][HI];
public:
	OneNetwork();
	void train(float inputs[IN], float targets[OU]);
	void query(float inputs[IN], float targets[OU]);
};
