#ifndef _IRIS_H_  
#define _IRIS_H_  

#include <vector>  
#include "bp.hpp"
class IRIS
{
public:
	IRIS(){bp=new BP();}
	void ReadFile(const char * InutFileName, int m, int n);
	void ReadTestFile(const char * InputFileName, int m, int n);
	void WriteToFile(const char * OutPutFileName);
    void TrainBP(Net *);
    Vector<Type> ForeCast(const Vector<Type>);/*test bp with input data*/
    void ForCastFromFile();/* test bp with test.data from iris dataset.*/
private:
	void SetData();
    void SetTestData();
	void split(char *buffer, Vector<std::string> &vec);
	void SplitString(const std::string& s, Vector<std::string>& v, const std::string& c);
private:
    Vector<Vector<Type> > result;  /* test results */
	Vector<Data> data;            /* sample data */
	Vector<Data> testdata; /* test data */
private:

	BP *bp;
};

#endif  //_IRIS_H_  