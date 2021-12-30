#include <chrono>
#include <random>
#include "OneNetwork.h"

using namespace std;

OneNetwork::OneNetwork()
{
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	minstd_rand0 generator(seed);
	for (int i = 0; i < HI; i++) {
		for (int j = 0; j < IN; j++) {
			wInputHidden[i][j] = (float)generator() / generator.max() - 0.5f;
		}
	}
	for (int i = 0; i < OU; i++) {
		for (int j = 0; j < HI; j++) {
			wHiddenOutput[i][j] = (float)generator() / generator.max() - 0.5f;
		}
	}
}

void OneNetwork::train(float in[IN], float targ[OU])
{
	float inputs[IN][1];
	float targets[OU][1];
	for (int i = 0; i < IN; i++)
		inputs[i][0] = in[i];
	for (int i = 0; i < OU; i++)
		targets[i][0] = targ[i];

	// Dot product:  hiddenInputs[HI][1] = wInputHidden[HI][IN] * inputs[IN][1]
	// Element-wise: hiddenOutputs = activation(hiddenInputs)
	float hiddenInputs[HI][1];
	float hiddenOutputs[HI][1];
	for (int row = 0; row < HI; row++) {
		hiddenInputs[row][0] = 0.0;
		for (int col = 0; col < IN; col++) {
			hiddenInputs[row][0] += wInputHidden[row][col] * inputs[col][0];
			hiddenOutputs[row][0] = 1.0f / (1.0f + exp((-1.0f) * hiddenInputs[row][0]));
		}
	}
	// Dot product:  finalInputs[HI][1] = wHiddenOutput[HI][IN] * hiddenOutputs[IN][1]
	// Element-wise: finalOutputs = activation(hiddenOutputs)
	float finalInputs[OU][1];
	float finalOutputs[OU][1];
	for (int row = 0; row < OU; row++) {
		finalInputs[row][0] = 0.0f;
		for (int col = 0; col < HI; col++) {
			finalInputs[row][0] += wHiddenOutput[row][col] * hiddenOutputs[col][0];
			finalOutputs[row][0] = 1.0f / (1.0f + exp((-1.0f) * finalInputs[row][0]));
		}
	}
	// Element-wise: outputError = targets - finalOutputs
	float outputErrors[OU][1];
	for (int i = 0; i < OU; i++) {
		outputErrors[i][0] = targets[i][0] - finalOutputs[i][0];
	}
	// Transpose weight matrix wHiddenOutput -> wHiddenOutputT
	float wHiddenOutputT[HI][OU];
	for (int row = 0; row < OU; row++) {
		for (int col = 0; col < HI; col++) {
			wHiddenOutputT[col][row] = wHiddenOutput[row][col];
		}
	}
	// Dot product: hiddenErrors[HI][1] = wHiddenOutputT[HI][OU] * outputErrors[OU][1]
	float hiddenErrors[HI][1];
	for (int row = 0; row < HI; row++) {
		hiddenErrors[row][0] = 0.0f;
		for (int col = 0; col < OU; col++) {
			hiddenErrors[row][0] += wHiddenOutputT[row][col] * outputErrors[col][0];
		}
	}
	// Element-wise: outMatrix = outputErrors * finalOutputs * (1.0 - finalOutputs)
	float outMatrix[OU][1];
	for (int i = 0; i < OU; i++) {
		outMatrix[i][0] = outputErrors[i][0] * finalOutputs[i][0] * (1.0f - finalOutputs[i][0]);
	}
	// Element-wise: inMatrix = hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)
	float inMatrix[HI][1];
	for (int j = 0; j < HI; j++) {
		inMatrix[j][0] = hiddenErrors[j][0] * hiddenOutputs[j][0] * (1.0 - hiddenOutputs[j][0]);
	}
	// Transpose matrix hiddenOutputs -> hiddenOutputsT
	float hiddenOutputsT[1][HI];
	for (int i = 0; i < HI; i++)
		hiddenOutputsT[0][i] = hiddenOutputs[i][0];

	// Dot product: dwHiddenOutput[OU][HI] = outMatrix[OU][1] * hiddenOutputsT[1][HI]
	float dwHiddenOutput[OU][HI];
	for (int row = 0; row < OU; row++) {
		for (int col = 0; col < HI; col++) {
			dwHiddenOutput[row][col] = outMatrix[row][0] * hiddenOutputsT[0][col];
		}
	}
	// Element-wise: update weight matrix elements by learnrate * delta
	for (int row = 0; row < OU; row++) {
		for (int col = 0; col < HI; col++) {
			wHiddenOutput[row][col] += LR * dwHiddenOutput[row][col];
		}
	}
	// Transpose matrix inputs -> inputsT
	float inputsT[1][IN];
	for (int j = 0; j < IN; j++)
		inputsT[0][j] = inputs[j][0];

	// Dot product: dwInputHidden[HI][IN] = inMatrix[HI][1] * inputsT[1][IN]
	float dwInputHidden[HI][IN];
	for (int row = 0; row < HI; row++) {
		for (int col = 0; col < IN; col++) {
			dwInputHidden[row][col] = inMatrix[row][0] * inputsT[0][col];
		}
	}
	// Element-wise: update weight matrix elements by learnrate * delta
	for (int row = 0; row < HI; row++) {
		for (int col = 0; col < IN; col++) {
			wInputHidden[row][col] += LR * dwInputHidden[row][col];
		}
	}
}

void OneNetwork::query(float in[IN], float targ[OU])
{
	// Create column vectors from input data lists
	double inputs[IN][1];
	for (int i = 0; i < IN; i++)
		inputs[i][0] = in[i];

	// Dot product:  hiddenInputs[HI][1] = wInputHidden[HI][IN] * inputs[IN][1]
	// Element-wise: hiddenOutputs = activation(hiddenInputs)
	double hiddenInputs[HI][1];
	double hiddenOutputs[HI][1];
	for (int row = 0; row < HI; row++) {
		hiddenInputs[row][0] = 0.0;
		for (int col = 0; col < IN; col++) {
			hiddenInputs[row][0] += wInputHidden[row][col] * inputs[col][0];
			hiddenOutputs[row][0] = 1.0 / (1.0 + exp((-1.0) * hiddenInputs[row][0]));
		}
	}
	// Dot product:  finalInputs[HI][1] = wHiddenOutput[HI][IN] * hiddenOutputs[IN][1]
	// Element-wise: finalOutputs = activation(hiddenOutputs)
	double finalInputs[OU][1];
	double finalOutputs[OU][1];
	for (int row = 0; row < OU; row++) {
		finalInputs[row][0] = 0.0;
		for (int col = 0; col < HI; col++) {
			finalInputs[row][0] += wHiddenOutput[row][col] * hiddenOutputs[col][0];
			finalOutputs[row][0] = 1.0 / (1.0 + exp((-1) * finalInputs[row][0]));
		}
	}
	// Write values of finalOutputs column vector to result list
	for (int i = 0; i < OU; i++) {
		targ[i] = finalOutputs[i][0];
	}
}
