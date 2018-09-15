#ifndef _BP_H_  
#define _BP_H_  

#include <vector>  

/* parameters settting */
#define LAYER    10      /* number of layers neutral network */ 
#define NUM      128       /* maxium notes in each layer */ 

#define A        1.0  
#define B        1.0    /* A, B sigmoid function parameters */ 
#define ITERS    1500     /* maximum training iterations*/  
#define ETA_W    0.035   /* weight rate */ 
#define ETA_B    0.025    /* bias rate */ 
#define ERROR    0.1    /* single sample error */  
#define ACCU     0.005    /* iteration error */ 

/* data type */
#define Type double  
#define Vector std::vector  

struct Data
{
    Vector<Type> x;       /* input type */
    Vector<Type> y;       /* output type*/
    Vector<std::string> tag;     /* tag info*/ 
};

struct Net
{
   int layerNum;
   Vector <int> nodeNum;
};

class BP{

public:

    void GetData(const Vector<Data>);
	void GetTestData(const  Vector<Data>);
	void GetRestRowLen(const  int _rowLen);
	void GetRowLen(const  int _rowLen);
    void Train(Net *);
    Vector<Type> ForeCast(const Vector<Type>);
	void ForCastFromFile(BP * &, Vector<Vector<Type> >&);

private:

    void InitNetWork();         /* initalize network */  
    void GetNums(Net *);             /* obtain the input, output and hidden layers notes */
    void ForwardTransfer();     /* forward transfer */  
    void ReverseTransfer(int);  /* backward transfer */  
    void CalcDelta(int);        /* caculate w and b */ 
    void UpdateNetWork();       /* update weights the threshold */
    Type GetError(int);         /* caculate sample error */ 
    Type GetAccu();             /* caculate the accuracy */  
    Type Sigmoid(const Type);   /* caculate sigmoid value */ 
    Type CostCross();

private:
    int in_num;                 /* input layer notes */  
    int ou_num;                 /* output layter notes */  
    Net nn;                     /* net structal */  

    Vector<Data> data;     /* sample data */
    Vector<Data> testdata; /* test data */
    int rowLen;            /* sample num */
    int restrowLen;        /* test num */

    Type w[LAYER][NUM][NUM];    /* network weights */ 
    Type b[LAYER][NUM];         /* network threshold */ 

    Type x[LAYER][NUM];         /* output value of each layer */
    Type d[LAYER][NUM];         /* save all deltas Wij(t+1)=Wij(t)+α(Yj-Aj(t))Oi(t) */
    int neurons[LAYER];
};

#endif  //_BP_H_  