#include <string.h>  
#include <stdio.h>  
#include <math.h>  
#include <assert.h>  
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "iris.hpp"
using namespace std;
/* prediction using input data*/
Vector<Type> IRIS::ForeCast(const Vector<Type> in)
{
  Vector<Type>  out;
  return out=bp->ForeCast(in);	
}
/* prediction using iris dataset*/
void IRIS::ForCastFromFile()
{
  bp->ForCastFromFile(bp, result);
}
/* trianing for iris*/
void IRIS::TrainBP(Net *net)
{
	bp->Train(net);
}

void IRIS::split(char *buffer, Vector<std::string> &vec)
{
	char *p = strtok(buffer, " ,");      //\t
	vec.push_back(p);
	while (p != NULL)
	{
		p = strtok(NULL, ",");
		if (p != NULL)
			vec.push_back(p);
	}
}
/* split data from string */
void IRIS::SplitString(const std::string& s, Vector<std::string>& v, const std::string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}
/* prepare data for bp train*/
void IRIS::SetData()
{
   	bp->GetData(data);
	bp->GetRowLen(data.size());
}
/* prepare data for bp test*/
void IRIS::SetTestData()
{
	bp->GetTestData(testdata);
	bp->GetRestRowLen(testdata.size());
}
/*read iris dataset*/
void IRIS::ReadFile(const char * InutFileName, int m, int n)
{
	ifstream fin;
	fin.open(InutFileName);

	if (!fin)
	{
		printf("open file %s failed...\n", InutFileName);
		exit(0);
	}

	/* init dataSet */
	Vector<string> temp;
    std::string line;
	while (getline(fin, line))//(fgets(buffer, 1024, pFile))
	{
		Data t;
		temp.clear();
		SplitString(line, temp, ",");
		for (int i = 0; i < m; i++)
		{
			Type dd;
			sscanf(temp[i].c_str(), "%lf", &dd);
		    t.x.push_back(dd);	
		}
        t.tag.push_back(temp[m]);
        if (temp[m].compare("Iris-setosa") == 0)
        {
        	t.y.push_back(1);
        	t.y.push_back(0);
        	t.y.push_back(0);
        }
        else if (temp[m].compare("Iris-versicolor") == 0)
        {
        	t.y.push_back(0);
            t.y.push_back(1);
            t.y.push_back(0);
        }
        else if (temp[m].compare("Iris-virginica") == 0)
        {
        	t.y.push_back(0);
            t.y.push_back(0);
            t.y.push_back(1);
        }
		data.push_back(t);
	}
	SetData();
	fin.close();
}
/*read iris dataset*/
void IRIS::ReadTestFile(const char * InputFileName, int m, int n)
{
	ifstream fin;
	fin.open(InputFileName);
	if (!fin)
	{
		printf("open file %s failed...\n", InputFileName);
		exit(0);
	}

	//init dataSet  
	Vector<std::string> temp;
    std::string line;
	while (getline(fin, line))
	{
		Data t;
		temp.clear();
		SplitString(line, temp, ",");
		for (int i = 0; i < m; i++)
		{
			Type dd;
			sscanf(temp[i].c_str(), "%lf", &dd);
			t.x.push_back(dd);
		}
		t.tag.push_back(temp[m]);
		testdata.push_back(t);
	}
	SetTestData();
	fin.close();
}
/*save prediction result*/
void IRIS::WriteToFile(const char * OutPutFileName)
{
	ofstream fout;
	fout.open(OutPutFileName);
	fout << "No."<< ","<<"0"<< ","<<"1"<< ","<<"2"<< ","<<"3"<< ",";
	fout << "tag"<< ","<< "predict"<<endl;
	if (!fout)
	{
		cout << "file result.txt  open failed" << endl;
		exit(0);
	}

	Vector<Data> ::iterator it = testdata.begin();
	Vector<Vector<Type> > ::iterator itr = result.begin();
	int dataIdx=0;
	while (it != testdata.end())
	{
		Data itt = (*it);
		Vector<Type> ::iterator ittx = itt.x.begin();
		Vector<std::string> ::iterator itt_tag = itt.tag.begin();
		Vector<Type> ::iterator ittr = (*itr).begin();
        fout << dataIdx << ",";
        dataIdx++;
		while (ittx != (itt.x).end())
		{
			fout << (*ittx) << ",";
			ittx++;
		}
		

		while (itt_tag != (itt.tag).end())
		{
			fout << (*itt_tag) << ",";
			itt_tag++;
		}
		
		double max = -1;
        int index = -1;
		while (ittr != (*itr).end())
		{
			if (max <= (*ittr))
            {
                max = (*ittr);
                index++;
            }
			ittr++;
		}

		switch (index)
        {
        case 0:
            fout << "Iris-setosa";
            break;
        case 1:
            fout << "Iris-versicolor";
            break;
        case 2:
            fout << "Iris-virginica";
            break;
        default:
            break;
        }
		it++;
		itr++;
		fout << "\n";
	}
	fout.close();
}
