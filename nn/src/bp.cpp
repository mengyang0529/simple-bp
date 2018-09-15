#include <string.h>  
#include <stdio.h>  
#include <math.h>  
#include <assert.h>  
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
using namespace std;

#include "bp.hpp"  

/* obtain all the sample data */ 
void BP::GetData(const Vector<Data> _data)
{
    data = _data;
}

/* obtain all the test data */
void BP::GetTestData(const  Vector<Data>  _testdata)
{
	testdata = _testdata;
}

/* obtain all the rowlen data */
void BP::GetRestRowLen(const  int _restrowLen)
{
	restrowLen = _restrowLen;
}

/* obtain all the rowlen data */
void BP::GetRowLen(const  int _rowLen)
{
	rowLen = _rowLen;
}

/* start training */ 
void BP::Train(Net *net)
{
    printf("Begin to train BP NetWork!\n");
    GetNums(net);
    InitNetWork();
    int num = data.size();

    for (int iter = 0; iter <= ITERS; iter++)
    {
        for (int cnt = 0; cnt < num; cnt++)
        {
            /* first layer values */
            for (int i = 0; i < in_num; i++)
                x[0][i] = data.at(cnt).x[i];

            while (1)
            {
                ForwardTransfer();
               
                if (GetError(cnt) < ERROR) {
                    break;
                }                   
                ReverseTransfer(cnt);
            }
        }
        printf("[ITER:%d]  ", iter);
        Type cost = CostCross();
        Type accu = GetAccu();               /* accuracy 1.0 -  err */
        printf("accuracy = %.5f,cost = %.5f\n", accu,cost);
        if (accu < ACCU) break;
    }
    printf("The BP NetWork train End!\n");
}

/* predict using trained network */ 
Vector<Type> BP::ForeCast(const Vector<Type> data)
{
    int n = data.size();
  //  assert(n == in_num);
    for (int i = 0; i < in_num; i++)
        x[0][i] = data[i];

    ForwardTransfer();
    int oInd=nn.layerNum - 1;
    Vector<Type> v;
    for (int i = 0; i < ou_num; i++)
        v.push_back(x[oInd][i]);
    return v;
}
/* prediction */
void  BP::ForCastFromFile(BP * &pBp, Vector<Vector<Type> > &result)
{   

    Vector<Data> ::iterator it = testdata.begin();
    Vector<Type> ou;
    while (it != testdata.end())
    {
        Vector<Type> itt=(*it).x;

        ou = pBp->ForeCast(itt);
        result.push_back(ou);

        it++;
    }
}

/* get network notes */  
void BP::GetNums(Net *net)
{

    if(net!=NULL)
    {   
        nn.layerNum=net->layerNum;

        in_num = data[0].x.size();                         /* get input layer notes num */ 
        nn.nodeNum.push_back(in_num);  
    
        for(int i=1;i<nn.layerNum-1;i++)                   /* get hidden layer notes num */ 
        {
            int hdNum=net->nodeNum[i];
            if (hdNum  > NUM) hdNum  = NUM;                /* hidden layers num no larger than the maximum */           
            nn.nodeNum.push_back(hdNum);  
        }

        ou_num = data[0].y.size();                         /* get output layer notes num */ 
        nn.nodeNum.push_back(in_num);  
    }
    else
    {   
        nn.layerNum=3;   
        in_num = data[0].x.size();                         /* get input layer notes num */ 
        nn.nodeNum.push_back(in_num); 
        
        int hdNum=(int)sqrt((in_num + ou_num) * 1.0) + 5;  /* get hidden layers notes num */   
        if (hdNum  > NUM) hdNum = NUM;                     /* hidden layers num no larger than the maximum */
        nn.nodeNum.push_back(hdNum);  
        
        ou_num = data[0].y.size();                         /* get output layer notes num */ 
        nn.nodeNum.push_back(in_num);  
    }
}

/* init network */
void BP::InitNetWork()
{
    //memset(w, 0, sizeof(w));      /* init weight and threshold */  
    memset(b, 0, sizeof(b));
    
    for (int i = 0; i < nn.layerNum; i++) {
        for (int j = 0; j < nn.nodeNum[i]; j++) {
			b[i][j] = rand() / double(RAND_MAX);
            for (int k = 0; k < NUM; k++) {
                w[i][j][k] = rand() / double(RAND_MAX);
            }
        }       
    }
}

/* forward */ 
void BP::ForwardTransfer()
{
    /* caculate output of every layers */  
    for(int k = 1;k < nn.layerNum;k++)
    {
        for (int j = 0; j < nn.nodeNum[k]; j++)
        {
            Type t = 0;
            for (int i = 0; i < nn.nodeNum[k-1]; i++) {
                t += w[k][i][j] * x[k-1][i];
            }
            t += b[k][j];
            x[k][j] = Sigmoid(t);
        }
    }
}

/* cacluate error */  
Type BP::GetError(int cnt)
{
    Type ans = 0;
    int oInd=nn.layerNum - 1;
    for (int i = 0; i < ou_num; i++)
        ans += 0.5 * (x[oInd][i] - data.at(cnt).y[i]) * (x[oInd][i] - data.at(cnt).y[i]); 
    return ans;
}

/* backpropagation */  
void BP::ReverseTransfer(int cnt)
{
    CalcDelta(cnt);
    UpdateNetWork();
}

/* cacluate accuacy */ 
Type BP::GetAccu()
{
    Type ans = 0;
    int num = data.size();
    int oInd=nn.layerNum - 1;
    for (int i = 0; i < num; i++)
    {
        int m = data.at(i).x.size();
        for (int j = 0; j < m; j++)
            x[0][j] = data.at(i).x[j];
        ForwardTransfer();
        int n = data.at(i).y.size();

        double max = -1;
        int index = -1;
        int indexD=-1;
        for (int j = 0; j < n; j++)
        {
            if(x[oInd][j]>max)
            {
                max=x[oInd][j];
                index=j;
            }
            if(data.at(i).y[j]==1)
            {
                indexD=j;
            }
        }
        if(index==indexD)
            ans++;

    }
    return ans / num;
}

/* calculate cross entropy*/
Type BP::CostCross()
{
    Type sum = 0;
    int num = data.size();
    int oInd=nn.layerNum - 1;
    for (int i = 0; i < num; i++)
    {
        int m = data.at(i).x.size();
        for (int j = 0; j < m; j++)
            x[0][j] = data.at(i).x[j];
        ForwardTransfer();
        int n = data.at(i).y.size();
        for (int j = 0; j < n; j++)
        {
           Type a = x[oInd][j];
           Type y = data.at(i).y[j];
            /* cross entropy */
            sum += -(y * log(a) + (1 - y) * log(1 - a));
        }

    }
    return sum/num;
}

/* caculate delta */ 
void BP::CalcDelta(int cnt)
{
    /* output layer delta */ 
    int oInd=nn.layerNum - 1;
    for (int i = 0; i < ou_num; i++)
        d[oInd][i] = (x[oInd][i] - data.at(cnt).y[i]) * x[oInd][i] * (A - x[oInd][i]) / (A * B);
    /* hidden layer delta */ 
    for(int k=nn.layerNum-2;k>0;k--)
    {
        for (int i = 0; i < nn.nodeNum[k]; i++)
        {
            Type t = 0;
            for (int j = 0; j < nn.nodeNum[k-1]; j++)
                t += w[k+1][i][j] * d[k+1][j];
           
            d[k][i] = t * x[k][i] * (A - x[k][i]) / (A * B);
        }
    }
}

/* update network according delta */ 
void BP::UpdateNetWork()
{
    int oInd=nn.layerNum - 1;

    for(int k=1;k<nn.layerNum;k++)
    {
        for (int i = 0; i < nn.nodeNum[k-1]; i++)
        {
            for (int j = 0; j < nn.nodeNum[k]; j++)
                w[k][i][j] -= ETA_W * d[k][j] * x[k-1][i];
        }
    }

    for(int k=1;k<nn.layerNum;k++)
    {
        for (int i = 0; i < nn.nodeNum[k]; i++)
            b[k][i] -= ETA_B * d[k][i];
    }
}

/* Sigmoid value */  
Type BP::Sigmoid(const Type x)
{
    return A / (1 + exp(-x / B));
}