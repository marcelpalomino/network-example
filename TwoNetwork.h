#pragma once

#define IN 1024
#define HI 250
#define OU 50
#define LA 10
#define LR 0.3

class TwoNetwork
{
private:
	float wInputHidden[HI][IN];
	float wHiddenOutput[OU][HI];
	float wOutputLast[LA][OU];
public:
	TwoNetwork();
	void train(float inputs[IN], float targets[LA]);
	void query(float inputs[IN], float targets[LA]);
};
