
#include <iostream>
#include <vector>

#include "CNN.h"
#include "Volumes.h"
using namespace std;


bool        adam;
double      bias, eta;
vector<int> image_1{1,28,28}, kernels_1{8,3,3,1};
vector<int> image_2{8,13,13}, kernels_2{2,3,3,8},  hidden{72};
int         input_layer, num_classes, epochs, padding, stride;


int main(int argc, char ** argv){

    //network istantiation

    CNN network;

    //build the network 
    cout << "Building network..." << endl;
    network.add_conv(image_1, kernels_1, padding= 0, stride= 2, bias= 0.1, eta= 0.01 );
    cout << "Added first conv layer" << endl;
    network.add_conv(image_2 , kernels_2 , padding= 0, stride= 2, bias= 0.1, eta= 0.01);
    cout << "Added second conv layer" << endl;
    network.add_dense(input_layer=2*6*6, hidden, num_classes=10, bias=1.0,  adam=false, eta=0.5);
    cout << "Added dense layer" << endl;

    //load the wanted dataset
    cout << "Loading MNIST dataset..." << endl;
    network.load_dataset("MNIST");
    cout << "Dataset loaded successfully" << endl;

    //sanity check

    //network.sanity_check();

    //train the network (Batch Size = 1)

    network.training(epochs=1, 10);

    //evaluate new samples 

    network.testing(10);


    return 0;

}
