#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char **argv)
{
	string file_name;
	if(argc == 2)
	{
		file_name = string(argv[1]);
	}
	else
	{
		printf("usage:./partition file_name");
		exit(0);
	}


	const int file_number = 16;

	FILE*fp = fopen(file_name.c_str(), "rb");
	//get file size
	fseek(fp, 0, SEEK_END); // seek to end of file
	long long nnz = ftell(fp)/12; // get current file pointer
	fseek(fp, 0, SEEK_SET); // seek back to beginning of file
	long long partition_size = (nnz + file_number - 1)/file_number;


	//open file
	FILE *f_output[file_number];
	for(int i = 0;i < file_number; i++)
	{
		stringstream ss;
		ss << i;
		string f_output_name = file_name + ss.str();
		cout << f_output_name << endl;

		f_output[i] = fopen(f_output_name.c_str(), "wb");
	}

	//output
	for(long long i = 0;i < nnz; i++)
	{
		int u,v;
		float rate;

		fread(&u, sizeof(int), 1, fp);
		fread(&v, sizeof(int), 1, fp);
		fread(&rate, sizeof(float), 1, fp);

		fwrite(&u, sizeof(int), 1, f_output[i/partition_size]);
		fwrite(&v, sizeof(int), 1, f_output[i/partition_size]);
		fwrite(&rate, sizeof(int), 1, f_output[i/partition_size]);
		

		if(i%10000000 == 0)printf("progress:%2lld, %%%.2f\n", i/partition_size, i%partition_size*100.0/partition_size);
	}

	fclose(fp);
	for(int i = 0;i < file_number; i++)fclose(f_output[i]);

	return 0;
}