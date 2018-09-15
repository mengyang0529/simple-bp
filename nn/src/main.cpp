#include <iostream>  
#include <string.h>  
#include <stdio.h>  
#include <math.h>
using namespace std;

#include "iris.hpp"  

int main()
{
    unsigned int Id, Od;    /* input, output dimensions */
    char select;
	IRIS *iris = new IRIS();
	Net net;

    const char * inputDataName = "../data/iris/train.data";/* trainging data file */
    const char * testDataName = "../data/iris/test.data";   /* test data file */
    const char * outputDataName = "result.csv";     /* output file */

    /*printf("please input sample input dimension and output dimension:\n");
    scanf("%d%d", &Id, &Od);*/
    Id = 4;
    Od = 3;
	
	iris->ReadFile(inputDataName, Id, Od);
    /* set layers*/
    net.layerNum=3;

    net.nodeNum.push_back(Id);
    int hdNum=(int)sqrt((Id + Od) * 1.0) + 5;
    net.nodeNum.push_back(hdNum);
    net.nodeNum.push_back(Od);

    /* trainging */
	iris->TrainBP(&net);
    //Test
    printf("\n******************************************************\n");
    printf("*press enter to start test\n");
    printf("********************************************************\n");
    scanf("%c", &select);

    iris->ReadTestFile(testDataName, Id, Od);
    /* Test */
    iris->ForCastFromFile();
    iris->WriteToFile(outputDataName);
    printf("the result have been save in the file :result.csv.\n");


    delete iris;
    return 0;
}

static void ShowMessage(void)
{
    printf("\n Neural Network Sample\n"
        "\n"
        "Message: -l: train file \n"
        "-t: test file \n"
        "-v: validation file \n"
        "-r: result file \n"
        "-L: layer number\n"
        "3 - 10 (default 3)\n"
        "");
}