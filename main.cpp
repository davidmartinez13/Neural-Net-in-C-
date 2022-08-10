#include <iostream>
#include "neural_net.hpp"
#include <vector>
#include <cstdio>
using namespace std;
using namespace matrix;

void train (uint32_t epoch,SimpleNeuralNetwork &nn,
            vector<vector<float>> targetInputs,vector<vector<float>> targetOutputs){
    cout << "training start\n";

    for(uint32_t i = 0; i < epoch; i++)
    {
        uint32_t index = rand() % 4;
        nn.feedForward(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);
    }

    cout << "training complete\n";
}
void test(vector<vector<float>> targetInputs, SimpleNeuralNetwork &nn){
    for( vector<float> input : targetInputs)
    {
        nn.feedForward(input);
        vector<float> preds = nn.getPredictions();
        cout << input[0] << "," << input[1] <<" => " << preds[0] << endl;
    }
};

int main()
{
    // creating neural network
    // 2 input neurons, 3 hidden neurons and 1 output neuron 
    vector<uint32_t> topology = {2,3,1};
    SimpleNeuralNetwork nn(topology, 0.1);
    
    //sample dataset
    vector<vector<float>> targetInputs = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    }; 
    vector<vector<float>> targetOutputs = {
        {0.0f},
        {0.0f},
        {1.0f},
        {1.0f}
    };

    uint32_t epoch = 100000;
    
    //training the neural network with randomized data
    train(epoch, nn, targetInputs, targetOutputs);

    //testing the neural network
    test(targetInputs, nn);

    return 0;
}