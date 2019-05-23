// The training data is copied from https://github.com/mohaned-abid/male_female-classifier-machine-leaning-/blob/master/male_female_classifier_using_different_models.py, but the code is mine.

// This neural network predicts gender based on height, weight, and shoe size



#include <stdlib.h> 
#include <stdio.h>
#include <math.h> // M_E



// global variables
int i, num_weights = 3, num_inputs = 11;
double weight[3];

// function prototypes
double sigmoid(double x);
double derivative(double x);
double think(double x, int y);
void train(int iterations);
double normalize(double input, double max, double min);

// training data
double inputs[11][3] = {
    {181, 80, 44},
    {177, 70, 43},
    {160, 60, 38},
    {154, 54, 37},
    {166, 65, 40},
    {190, 90, 47},
    {175, 64, 39},
    {177, 70, 40},
    {159, 55, 37},
    {171, 75, 42},
    {181, 85, 43}
}; // height, weight, shoe size

int outputs[11] = {1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1}; // 0 = female, 1 = male



int main() {



// Normalize inputs for the sigmoid function.
double n0, n1, n2;
for (i=0; i<num_inputs; i++){
    n0 = normalize(inputs[i][0], 190, 154);
    inputs[i][0] = n0; // heights
    n1 = normalize(inputs[i][1], 90, 54);
    inputs[i][1] = n1; // weights
    n2 = normalize(inputs[i][2], 47, 37);
    inputs[i][2] = n2; // shoe size
    printf("%f %f %f %i\n", n0, n1, n2, outputs[i]);
    }


 
// Seed the pseudo-random number generator with the Process ID. No, this is not the best method. But, it works for this exercise.
srand((unsigned int)getpid()); 



// Generate random weights.
for (i=0; i<num_weights; i++){
    weight[i] = 2*(((double)rand())/RAND_MAX)-1;
    printf("weight%i %f\n", i, weight[i]);
    }



// Train the model using the training data.
train(200);



// Test the model with new inputs.
double t0, t1, t2;

n0 = normalize(160, 190, 154);
n1 = normalize(60, 90, 54);
n2 = normalize(40, 47, 37);
t0 = think(n0, 0);
t1 = think(n1, 1);
t2 = think(n2, 2);
printf("%f %f %f\n", n0, n1, n2);
printf("%f %f %f\n", t0, t1, t2);
printf("%f\n", normalize((t0 + t1 + t2), 3, 1.5));
if (normalize((t0 + t1 + t2), 3, 1.5) < .45){
    printf("The subject is probably female.\n");
} else if (normalize((t0 + t1 + t2), 3, 1.5) > .55){
    printf("The subject is probably male.\n\n");
} else {
    printf("The subject's gender cannot be determined.\n\n");
} // 170 70 43
    
n0 = normalize(185, 190, 154);
n1 = normalize(85, 90, 54);
n2 = normalize(45, 47, 37);
t0 = think(n0, 0);
t1 = think(n1, 1);
t2 = think(n2, 2);
printf("%f %f %f\n", n0, n1, n2);
printf("%f %f %f\n", t0, t1, t2);
printf("%f\n", normalize((t0 + t1 + t2), 3, 0));
if (normalize((t0 + t1 + t2), 3, 1.5) < .45){
    printf("The subject is probably female.\n");
} else if (normalize((t0 + t1 + t2), 3, 1.5) > .55){
    printf("The subject is probably male.\n\n");
} else {
    printf("The subject's gender cannot be determined.\n\n");
} // 170 70 43



    return 0;
}



// function definitions

// the Sigmoid normalizes weights between 0 and 1; it is graphed as an S-shaped curve
double sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

// indicates confidence in weights; Gradient Descent: lower confidence results in larger adjustments and zero values do not cause changes
double derivative(double x){
    return (x * (1 - x));
}

double result;
double think(double x, int y){
    result = sigmoid(x * weight[y]);
    return(result);
} 
        
void train(int iterations) {
    int j, k;
    double thought, error, adjustment;
    for (i=0; i<iterations; i++){
        for (j=0; j<num_inputs; j++){
            for (k=0; k<num_weights; k++){
            thought = think(inputs[j][k], k);
            error = (outputs[j] - thought);
            adjustment = (inputs[j][k] * error * derivative(thought));
            weight[k] += adjustment;
            }
        }
        if (i % 20 == 0) {
            printf("i %i error %f\n", i, error);
            for (k=0; k<num_weights; k++){
                printf("weight%i %f\n", k, weight[k]);
            }
            printf("\n");
        }
    }
}

double normalize(double input, double max, double min){
    return ((input - min) / (max - min));
}
