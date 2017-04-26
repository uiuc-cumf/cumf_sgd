#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char*argv[])
{

	if(argc != 2)
	{
		printf("usage: ./sample file_name\n");
		exit(0);
	}

	string filename(argv[1]);


	FILE*fp = fopen(filename.c_str(), "rb");

	string train_file_name = filename + string(".train");
	string test_file_name = filename + string(".test");

	FILE* f_train = fopen(train_file_name.c_str(), "wb");
	FILE* f_test = fopen(test_file_name.c_str(), "wb");

	fseek(fp, 0, SEEK_END); // seek to end of file
	long long file_size = ftell(fp); // get current file pointer
	fseek(fp, 0, SEEK_SET); // seek back to beginning of file

	long long nnz = file_size/12;

	cout <<nnz << endl;
	//return 0;


	srand(time(NULL));

	for(long long i = 0; i < nnz; i++)
	{

		if(i%10000000 == 0)printf("progress:%%%.3f\n",100.0*i/nnz);
		int u,v;
		float rate;

		fread(&u, sizeof(int), 1, fp);
		fread(&v, sizeof(int), 1, fp);
		fread(&rate, sizeof(float), 1, fp);

		double pos = double(rand()%1000);
		//cout << pos << endl;
		if( pos > 900)continue;
		else if(pos > 890)
		{
		
			fwrite(&u,sizeof(int),1,f_test);
			fwrite(&v,sizeof(int),1,f_test);
			fwrite(&rate, sizeof(float),1,f_test);
		}
		else
		{
			fwrite(&u,sizeof(int),1,f_train);
			fwrite(&v,sizeof(int),1,f_train);
			fwrite(&rate, sizeof(float),1,f_train);

		}
		
	}

	fclose(fp);
	fclose(f_train);
	fclose(f_test);


	return 0;

}