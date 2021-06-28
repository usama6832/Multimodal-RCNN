#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <sstream>
#include <algorithm>

//#include <omp.h>
#ifdef LINUX
#include <sys/time.h>
#else
#include <time.h>
#endif

using namespace std;
typedef vector<int> doc;

#include "fileutil.hpp"

//myself
//const int H=20;

//***************************************************************************************

const int H = 100; //Hidden layer
const int MAX_C = 50; //Maximum number of categories
const int MAX_F = 1500; //The maximum size of the input layer
const char *model_name = "model_300_nosuff_noinit";
double dropout = 0;
const char *train_file = "";
const char *valid_file = "";
const char *test_file = "";
const char *dict_file = "dict.txt";




//myself
//const int HH=100;
const int SROW=706;
const int SCOL=79;
const int HH=H+SCOL;
//const int HH=H;
double structdata[SROW][SCOL];
char *file_name="structData.txt";
//int trainCount=0;
//int testCount=0;
const int TRAINSUM=606;

int lastOutput[SROW-TRAINSUM];
int testLabel[SROW-TRAINSUM];

//***********************************************


int class_size; //Number of categories
int window_size; //Window size
int input_size; //Input size
int vector_size; //The vector size of a word unit = the word vector size (about 50) + the size of all features (about 10)

const int thread_num = 16;
//===================== All parameters to be optimized =====================


embedding_t words; //Word vector
embedding_t words_o; //Word vector

double *A; //Feature matrix: [number of categories][hidden layer] the weight of the second layer
double *B; //Feature matrix: [hidden layer][number of features] the weight of the first layer
double *gA, *gB;

double biasOutput[20]; //classsize
double h_b[H]; //The offset of the hidden layer

//===================== Known data =====================

//Training set
vector<doc> data; //Training data: [number of samples][number of features]
//int N; //training set size
//int uN; //unknown word
vector<int> b; //target matrix [number of samples] training set

//Validation set
vector<doc> vdata; //Test data: [number of samples][number of features]
//int vN; //Test set size
//int uvN; //unknown word
vector<int> vb; //target matrix [number of samples] test set

//Test set
vector<doc> tdata; //Test data: [number of samples][number of features]
//int tN; //Test set size
//int utN; //unknown word
vector<int> tb; //target matrix [number of samples] test set




double time_start;
double lambda = 0;// 0.01; //Regular item parameter weight
double alpha = 0.01; //learning rate
int iter = 0;

//myself reads txt structured data

int ReadStructureData(char *file_name)
{

    int i,j;
    for(i=0;i<SROW;i++)
		   for(j=0;j<SCOL;j++)
		      structdata[i][j]=0;


    FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("StructureData file not found\n");
		return -1;
	}


	while(!feof(f))
	{
	    for(i=0;i<SROW;i++)
		   for(j=0;j<SCOL;j++)
		      fscanf(f,"%lf",&structdata[i][j]);
	}

	fclose(f);

}
//**************************************************************************************


//====================================Visual inspection is to set the output time after each iteration====================
double getTime() {
#ifdef LINUX
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return 1.0*clock() / CLOCKS_PER_SEC;
#endif
}

//
double nextDouble() {
	return rand() / (RAND_MAX + 1.0);
}

//===================================softmax============================================
void softmax(double hoSums[], double result[], int n) {
	double max = hoSums[0];
	for (int i = 0; i < n; ++i)
	if (hoSums[i] > max) max = hoSums[i];
	double scale = 0.0;
	for (int i = 0; i < n; ++i)
		scale += exp(hoSums[i] - max);
	for (int i = 0; i < n; ++i)
		result[i] = exp(hoSums[i] - max) / scale;
}

//===================================sigmoid=============================================
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//==============================hardtanh========================================================
double hardtanh(double x) {
	if (x > 1)
		return 1;
	if (x < -1)
		return -1;
	return x;
}

//======================================================================================
//b = Ax  
void fastmult(double *A, double *x, double *b, int xlen, int blen) {              \\Recurrent Convolution
	double val1, val2, val3, val4;
	double val5, val6, val7, val8;
	int i;
	for (i = 0; i < blen / 8 * 8; i += 8) {
		val1 = 0;
		val2 = 0;
		val3 = 0;
		val4 = 0;

		val5 = 0;
		val6 = 0;
		val7 = 0;
		val8 = 0;

		for (int j = 0; j < xlen; j++) {                                                            
			val1 += x[j] * A[j + (i + 0)*xlen] + val8 * A[j + (i + 0)*xlen];
			val2 += x[j] * A[j + (i + 1)*xlen] + val1 * A[j + (i + 1)*xlen];
			val3 += x[j] * A[j + (i + 2)*xlen] + val2 * A[j + (i + 2)*xlen];
			val4 += x[j] * A[j + (i + 3)*xlen] + val3 * A[j + (i + 3)*xlen];

			val5 += x[j] * A[j + (i + 4)*xlen] + val4 * A[j + (i + 4)*xlen];
			val6 += x[j] * A[j + (i + 5)*xlen] + val5 * A[j + (i + 5)*xlen];
			val7 += x[j] * A[j + (i + 6)*xlen] + val6 * A[j + (i + 6)*xlen];
			val8 += x[j] * A[j + (i + 7)*xlen] + val7 * A[j + (i + 7)*xlen];
		}
		b[i + 0] += val1;
		b[i + 1] += val2;
		b[i + 2] += val3;
		b[i + 3] += val4;

		b[i + 4] += val5;
		b[i + 5] += val6;
		b[i + 6] += val7;
		b[i + 7] += val8;
	}

	for (; i < blen; i++) {
		for (int j = 0; j < xlen; j++) {
			b[i] += x[j] * A[j + i*xlen];
		}
	}
}

//=====================================================================================
//double checkCase(doc &id, int ans, int &correct, int &output, bool gd = false) {


//myself
double checkCase(doc &id, int ans, int &correct, int &output, int structCount, bool gd = false) {

//****************************************************************************************

	//	double x[MAX_F];
	int hw = (window_size - 1) / 2;
	double h[H];
	int maxhi[H];
	for (int j = 0; j < H; j++)  //To take max, initialize to -inf
		h[j] = -1e300;

	int start_word = 0;
	int end_word = (int)id.size();
	/*if (gd) {
		int sl = (int)(nextDouble()*nextDouble()* id.size() + 0.5);
		if (sl < 3)
		sl = 3;
		if (sl >= end_word) { //size
		//Sentence is too short, judge the whole sentence
		} else {
		start_word = rand() % (end_word - sl + 1);
		end_word = start_word + sl;
		if (start_word + hw >= end_word - hw) {
		start_word = 0;
		end_word = (int)id.size();
		}
		}
		}*/

	//Convolution to the last hidden layer
	for (int i = start_word + hw; i < end_word - hw; i++) { //Center of each window (padding has been added before and after
		double th[H] = { 0 };
		double x[MAX_F]; //Input layer serial connection

		for (int k = i - hw, xp = 0; k <= i + hw; k++) { //Every element in the window
			int offset = id[k] * words.element_size;
			for (int j = 0; j < words.element_size; j++, xp++) {
				x[xp] = words.value[offset + j];
			}
		}

		fastmult(B, x, th, input_size, H); //TODO will be the size of H1

		for (int j = 0; j < H; j++) {
			if (th[j] > h[j]) {
				h[j] = th[j];
				maxhi[j] = i; //Mark the maximum position
			}
		}
	}


	for (int i = 0; i < H; i++) {
		h[i] = max(0.0, h[i] + h_b[i]);
	}

   

    //myself
    double hh[HH];
    for (int j = 0; j < HH; j++){
		if(j<H){
			hh[j]=h[j];
		}else{

			hh[j] = structdata[structCount][j-H];
			/*
			if(gd){
				hh[j] = structdata[trainCount][j-H];
			}else{
				//hh[j] = structdata[testCount][j-H];

			}
			*/
			
		}
	}
	
	/*	
	if(gd){
		trainCount++;
	}else{
		testCount++;
	}
	*/	
	
//*******************************************************************




	double r[MAX_C] = { 0 };
	/*if (gd) { //During training
		for (int j = 0; j < H; j++) {
			if (nextDouble() < dropout) {
				h[j] = 0;
			}
		}
		for (int i = 0; i < class_size; i++) {
			r[i] = biasOutput[i];
			for (int j = 0; j < H; j++) {
				r[i] += h[j] * A[i*H + j];
			}
		}
	} else {
		for (int i = 0; i < class_size; i++) {
			r[i] = biasOutput[i];
			for (int j = 0; j < H; j++) {
				r[i] += h[j] * A[i*H + j] / (1 - dropout);
			}
		}
	}*/





//myself
if (gd) { //During training
		for (int j = 0; j < HH; j++) {
			if (nextDouble() < dropout) {
				hh[j] = 0;
			}
		}
		for (int i = 0; i < class_size; i++) {
			r[i] = biasOutput[i];
			for (int j = 0; j < HH; j++) {
				r[i] += hh[j] * A[i*HH + j];
			}
		}
	} else {
		for (int i = 0; i < class_size; i++) {
			r[i] = biasOutput[i];
			for (int j = 0; j < HH; j++) {
				r[i] += hh[j] * A[i*HH + j] / (1 - dropout);
			}
		}
	}

	//*******************************************************************************








	double y[MAX_C];
	softmax(r, y, class_size);

	double dy = ans - r[0];

    //����������������������������������������������������������������������������������������������
	if (gd) { //Change parameters
		double dh[H] = { 0 };
		if (class_size == 1) { //Return mission
			biasOutput[0] += alpha * (dy - lambda*biasOutput[0]);
			for (int j = 0; j < H; j++) {
				dh[j] += dy * A[j];
				//dh[j] *= 1 - h[j] * h[j];
				if (h[j] <= 0)
					dh[j] = 0;
			}
		} else {
			for (int i = 0; i < class_size; i++) {
				if (i == ans) {
					biasOutput[i] += alpha*(1 - y[i] - lambda*biasOutput[i]);
				} else {
					biasOutput[i] += alpha*(0 - y[i] - lambda*biasOutput[i]);
				}
			}
			for (int j = 0; j < H; j++) {
				dh[j] = A[ans*H + j];
				for (int i = 0; i < class_size; i++) {
					dh[j] -= y[i] * A[i*H + j];
				}
				//dh[j] *= 1 - h[j] * h[j];
				if (h[j] <= 0)
					dh[j] = 0;

				if (dropout > 0 && h[j] == 0)
					dh[j] = 0;
				//dh[j] *= h[j]*(1-h[j]);
				/*if(h[j] > 1 || h[j] < -1)
				dh[j] = 0;
				biasH[j] += alpha * dh[j];*/
				h_b[j] += alpha * dh[j];
			}
		}

		//#pragma omp critical
		{
			for (int i = 0; i < class_size; i++) {
				double v = (i == ans ? 1 : 0) - y[i];
				/*for (int j = 0; j < H; j++) {
					int t = i*H + j;
					A[t] += alpha / sqrt(H) * (v * h[j] - lambda * A[t]);
					//gA[i*H+j] += v * h[j];
				}*/


				//myself
				for (int j = 0; j < HH; j++) {
					int t = i*HH + j;
					A[t] += alpha / sqrt(HH) * (v * hh[j] - lambda * A[t]);
					//gA[i*H+j] += v * h[j];
				}

				//*************************************************************************************************


			}





			double dx[MAX_F] = { 0 };

			//fastmult(B, dh, dx, input_size, H);
			for (int i = 0; i < H; i++) {
				int p = maxhi[i];
				for (int k = p - hw, xp = 0; k <= p + hw; k++) { //Every element in the window
					for (int j = 0; j < words.element_size; j++, xp++) {
						double dx = dh[i] * B[i*input_size + xp];
						int t = id[k] * words.element_size + j;
						words.value[t] += alpha * (dx - lambda * (words.value[t] - words_o.value[t]));
					}
				}
			}

			//TODO
			for (int i = 0; i < H; i++) {

				int p = maxhi[i];
				for (int k = p - hw, xp = 0; k <= p + hw; k++) { //Every element in the window
					for (int j = 0; j < words.element_size; j++, xp++) {
						int t = i*input_size + xp;
						int wt = id[k] * words.element_size + j;

						B[t] += alpha / sqrt(input_size) * (words.value[wt] * dh[i] - lambda * B[t]);

					}
				}


				//for (int j = 0; j <words.element_size; j++) {
				// int t = i*words.element_size + j;
				// int wt = id[maxhi[i]] * words.element_size + j; //Only the element at this position needs to pass the error
				// B[t] += alpha / sqrt(words.element_size) * (words.value[wt] * dh[i]-lambda * B[t]);
				// //gB[i*input_size+j] += -x[j] * dh[i];
				// //TODO is best to use the old value of words.value[wt]
				//}
			}

			/*
			for (int i = 0, j = 0; i < window_size; i++) {
			int offset = id[i].word * words.element_size;
			for (int k = 0; k < words.element_size; k++, j++) {
			int t = offset + k;
			words.value[t] += alpha * (dx[j] - lambda * words.value[t]);
			}
			}*/

		}
	}

	output = 0;
	double maxi = 0;
	bool ok = true;
	for (int i = 0; i < class_size; i++) {
		if (i != ans && y[i] >= y[ans])
			ok = false;
		if (y[i] > maxi) {
			maxi = y[i];
			output = i;
		}
		//if (p)
		//	p[i] = -log(y[i]);
	}

	if (ok)
		correct++;
	if (class_size == 1)
		return dy*dy;
	return log(y[ans]); //Calculated likelihood
}


//===============================writeFile=======================================
void writeFile(const char *name, double *A, int size) {
	FILE *fout = fopen(name, "wb");
	fwrite(A, sizeof(double), size, fout);
	fclose(fout);
}


//dataset--->char[]      train valid test
double checkSet(const char *dataset, vector<doc> &data, vector<int> &b, char *fname = NULL) {
	int N = (int)data.size();

	double ret = 0;
	int wordCorrect = 0; //Direct word accuracy


	int sc[20][3] = {{0} };
//[c][0] goal=answer=c
//[c][1] goal=c answer!=c recall rate
//[c][2] Goal!=c Answer=c Accuracy

//Thread parallel iteration
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)

	for (int s = 0; s < N; s++) {
		int tc = 0;
		int output;
		//double tv = checkCase(data[s], b[s], tc, s, output);

		//myself
		double tv =0;
		//train dataset
		if(dataset[0]=='t' && dataset[1]=='r'){
			tv = checkCase(data[s], b[s], tc, output, s);
		}
		//valid dataset
		else if(dataset[0]=='v' && dataset[1]=='a'){
			tv = checkCase(data[s], b[s], tc, output, s+TRAINSUM);
		}
		//test dataset
		else if(dataset[0]=='t' && dataset[1]=='e'){
			tv = checkCase(data[s], b[s], tc, output, s+TRAINSUM);
			lastOutput[s]=output;
			testLabel[s]=b[s];
		}
		
		//***************************************************************************************


#pragma omp critical
		{
			ret += tv;
			if (output == b[s]) {
				sc[b[s]][0]++;
			} else {
				sc[b[s]][1]++;
				sc[output][2]++;
			}
			//printf("%lf, ", ret);
			wordCorrect += tc;
		}
	}
	double ff = 0;
	for (int i = 0; i < class_size; i++) {
		//mysellf
		//printf("%d %d %d\n", sc[i][0], sc[i][1], sc[i][2]);

		//myself
		double acc=(sc[0][0]+sc[1][0])/(double)N;
		
		//*****************************************************************************************

		double p = sc[i][0] == 0 ? 0 : 1.0* sc[i][0] / (sc[i][0] + sc[i][2]);
		double r = sc[i][0] == 0 ? 0 : 1.0*sc[i][0] / (sc[i][0] + sc[i][1]);
		double f = sc[i][0] == 0 ? 0 : 2 * p * r / (p + r);
		ff += f;
                if (i==1) 
              //  printf(" p:%lf r=%lf   ", p,r); 

		//myself
		printf("%lf %lf %lf %lf\t",acc, p,r,f);
		//*****************************************************************************************

	}
	/*printf("%s:%lf(%.2lf%%,%.2lf%%), ", dataset, -ret / N,
		100.*wordCorrect / N, ff / class_size * 100);*/
	return -ret / N;
}


//=============================================================================================
//Check the correct rate and likelihood
//The return value is likelihood
double check() {
	double ret = 0;

	double ps = 0;
	int pnum = 0;
	for (int i = 0; i < class_size*H; i++, pnum++) {
		ps += A[i] * A[i];
	}
	for (int i = 0; i < H*words.element_size; i++, pnum++) {
		ps += B[i] * B[i];
	}
	for (int i = 0; i < words.size; i++, pnum++) {
		ps += words.value[i] * words.value[i];
	}

	/*sprintf(fname, "%s_A", model_name);
	writeFile(fname, A, class_size*H);
	sprintf(fname, "%s_B", model_name);
	writeFile(fname, B, H*input_size);
	sprintf(fname, "%s_w", model_name);
	writeFile(fname, words.value, words.size);
	*/

    //myself
	//printf("para: %lf, ", ps / pnum / 2);

	ret = checkSet("train", data, b);

	//myself
	//testCount=606;
	//***************************************************************

//	checkSet("valid", vdata, vb);
	//sprintf(fname, "%s_%d_output", model_name, iter);

	//myself
	//testCount=606;
	//***************************************************************

	checkSet("test", tdata, tb);

//	printf("time:%.1lf\n", getTime() - time_start);
	//myself
	printf("%.1lf\n",getTime() - time_start);	

	//***************************************************************
	
	fflush(stdout);

	double fret = ret + ps / pnum*lambda / 2;
	return fret;
}


//=======================================readFile====================
int readFile(const char *name, double *A, int size) {
	FILE *fin = fopen(name, "rb");
	if (!fin)
		return 0;
	int len = (int)fread(A, sizeof(double), size, fin);
	fclose(fin);
	return len;
}

//Find an average value of embedding for all documents in a collection=================================
void SimplifyData(vector<doc> &data) {
	for (size_t i = 0; i < data.size(); i++) {
		doc &d = data[i];
		sort(d.begin(), d.end());
		d.erase(unique(d.begin(), d.end()), d.end());
	}
}


//==================================================SimplifyDataWordCh======================================
void SimplifyDataWordCh(vector<doc> &data) {
	int total = 0;
	for (size_t i = 0; i < data.size(); i++) {
		doc &d = data[i];

		doc lst;
		for (int j = 0; j < (int)d.size(); j++) {
			WordCh wc((char*)vocab[d[j]].c_str());
			int cnt = 0;
			while (char *cc = wc.NextCh()) { //Add each word
				if (dict.count(cc)) {
					lst.push_back(dict[cc]);
					cnt++;
				}
			}
			if (cnt != 1) {
				lst.push_back(d[j]);
			}
		}

		d = lst;

		sort(d.begin(), d.end());
		d.erase(unique(d.begin(), d.end()), d.end());

		total += d.size();
	}
	printf("total=%d\n", total);
}

/*
Training set, test set, validation set vec<doc>
doc=vec<word>
word=(string)ch~ch~ch

parameter
Embedding Each word and character has a corresponding hash table

Evaluate doc, return classification (retain intermediate variables, used to calculate gradient)


*/

//====================================Find an average value of embedding for all documents in a collection==================================

//filling! ! ! ! Fill two units before and after! ! !
void AddPadding(vector<doc> &data) {
	/*for (size_t i = 0; i < data.size(); i++) {
	doc &d = data[i];
	sort(d.begin(), d.end());
	d.erase(unique(d.begin(), d.end()), d.end());
	}*/
	int hw = (window_size - 1) / 2;

	for (size_t i = 0; i < data.size(); i++) {
		doc &d = data[i];
		doc dd;
		dd.reserve(d.size() + 2 * hw);
		for (int i = 0; i < hw; i++)
			dd.push_back(0);
		for (size_t j = 0; j < d.size(); j++) {
			dd.push_back(d[j]);
		}
		for (int i = 0; i < hw; i++)
			dd.push_back(0);
		d = dd;
	}
}



//=====================================================main==============================================================
int main(int argc, char **argv) {
	if (argc < 5) {
		printf("Useage: ./ecnn w(null) train test class_size rand_seed mr(1) window(5) valid(90=90%%train) dropout(0.5)\n");
		return 0;
	}
	for (int i = 0; i < argc; i++) {
		printf("%s ", argv[i]);
	}
	printf("\n");
	//model_name = argv[2];

	train_file = argv[2];//training set
	test_file = argv[3];//Test set

	if (argc >= 10)
	dropout = atof(argv[9]);
	printf("dropout=%lf\n", dropout);

	printf("read embedding\n");


	window_size = atoi(argv[7]);
	class_size = atoi(argv[4]); //TODO category information can be automatically obtained from the data
if (class_size == 1) {//It is especially easy to parameter burst when returning, so this must be added
		lambda = 0.01;
	}
	srand(atoi(argv[5]));

	if (strcmp(argv[6], "0") == 0)
		valid_file = NULL;
	else
		valid_file = argv[6];//Authentication set

	//vector_size = 100;

	//	init(dict_file);

	//words.init(vector_size, chk.size());

	printf("read data\n");
	//readAllData(train_file, "Train", window_size, data, b, N, uN);
	//readAllData(valid_file, "Valid", window_size, vdata, vb, vN, uvN);
	//readAllData(test_file, "Test", window_size, tdata, tb, tN, utN);

	//input_size = window_size * vector_size;

	//printf("init. input(features):%d, hidden:%d, output(classes):%d, alpha:%lf, lambda:%.16lf\n", input_size, H, class_size, alpha, lambda);
	//printf("window_size:%d, vector_size:%d, vocab_size:%d, allwordsLen:%d, lineMax:%d\n", window_size, vector_size, words.element_num, allwordsLen, lineMax);
	//checkWordsExists();

	if (valid_file == NULL) {
		ReadAllFiles(train_file, test_file, atoi(argv[8]), NULL, argv[1], 0, words,
			data, b, vdata, vb, tdata, tb);
	} else {
		ReadAllFiles(train_file, test_file, -1, valid_file, argv[1], 0, words,
			data, b, vdata, vb, tdata, tb);
	}
	printf("H=%d\n", H);

    //myself
	ReadStructureData(file_name);

	//****************************************************************

	//printf("words.element_size=%d,words.element_num=%d,word.size=%d",words.element_size,words.element_num,word.size)

	/*for (int i = 0; i < words.size; i++) {
		words.value[i] = (nextDouble() - 0.5);
		}
		for (int i = 0; i < 20; i++) {
		biasOutput[i] = (nextDouble() - 0.5);
		}*/


	{
		printf("initialized with %s\n", argv[1]);
		double sum = 0;
		for (int i = 0; i < words.size; i++) {
			sum += words.value[i] * words.value[i];
		}
		sum = sqrt(sum / words.size * 12);

		//Normalized? ? ?
		for (int i = 0; i < words.size; i++) {
			words.value[i] /= sum;
		}
		//Randomly initialize embedding
		/*if (atoi(argv[7]) == 1) {
			for (int i = 0; i < words.size; i++) {
			words.value[i] = (nextDouble() - 0.5);
			}
			printf("rand initialized\n");
			}*/
	}
	//if (ReadEmbedding(words, argv[1]) != -1) {
	//	printf("initialized with %s\n", argv[1]);
	//	double sum = 0;
	//	for (int i = 0; i < words.size; i++) {
	//		sum += words.value[i] * words.value[i];
	//	}
	//	sum = sqrt(sum / words.size * 12);
	//	for (int i = 0; i < words.size; i++) {
	//		words.value[i] /= sum;
	//	}
	//	//Randomly initialize embedding
	//	//for (int i = 0; i < words.size; i++) {
	//	//	words.value[i] = (nextDouble() - 0.5);
	//	//}
	//	/*if (argc > 3) {
	//		double v = atof(argv[3]);
	//		printf("x%lf %s\n", v);
	//		for (int i = 0; i < words.size; i++) {
	//		words.value[i] *= v;
	//		}
	//		}*/
	//} else {
	//	printf("not initialized\n");
	//}

	//Transfer to words_o
	words_o = words;
	words_o.value = new double[words_o.size];
	memcpy(words_o.value, words.value, sizeof(double)*words.size);
	//lambda = 0.001;

    //Input size
	input_size = words.element_size*window_size;

	//A = new double[class_size*H];

	//myself
	A = new double[class_size*HH];
	//***********************************************************************************
	
	
	//gA = new double[class_size*H];
	B = new double[H*input_size];
	//gB = new double[H*input_size];

    //The roles of A and B
	/*for (int i = 0; i < class_size * H; i++) {
		A[i] = (nextDouble() - 0.5) / sqrt(H);
	}*/

	
	
	//myself
	for (int i = 0; i < class_size * HH; i++) {
		A[i] = (nextDouble() - 0.5) / sqrt(HH);
	}
	//***********************************************************************************





	for (int i = 0; i < H * input_size; i++) {
		B[i] = (nextDouble() - 0.5) / sqrt(input_size);
		//B[i] = (nextDouble()*0.02 - 0.01) ;
	}

	//readFile("ECCC_1", B, H*input_size);
	//printf("read data\n");
	//ReadDocs(train_file, data, b, "Train");
	//ReadDocs(test_file, tdata, tb, "Test");

	//SimplifyData(data);
	//SimplifyData(tdata);
	AddPadding(data);
	AddPadding(tdata);
	AddPadding(vdata);

   


	time_start = getTime();

	int N = data.size();

	int *order = new int[N];
	for (int i = 0; i < N; i++) {
		order[i] = i;
	}

	//srand(atoi(argv[5])); //Ensure that the data sets of the same seed segmentation are consistent
	//for (int i = 0; i < N; i++) {
	//	int p = rand() % N;
	//	swap(data[i], data[p]);
	//	swap(b[i], b[p]);
	//}

	//if (mr) {
	//	for (int i = N * 9 / 10; i < N; i++) {
	//		tdata.push_back(data[i]);
	//		tb.push_back(b[i]);
	//	}
	//	data.erase(data.begin() + N * 9 / 10, data.end());
	//	b.erase(b.begin() + N * 9 / 10, b.end());
	//	N = data.size();
	//} else {
	//	ReadDocs(test_file, tdata, tb, "Test");
	//	AddPadding(tdata);
	//}


	////Another verification set
	//int validStart = N * atoi(argv[8]) / 100;
	//for (int i = validStart; i < N; i++) {
	//	vdata.push_back(data[i]);
	//	vb.push_back(b[i]);
	//}
	//data.erase(data.begin() + validStart, data.end());
	//b.erase(b.begin() + validStart, b.end());
	//N = data.size();

	printf("%lu, %lu, %lu\n", data.size(), vdata.size(), tdata.size());
         

        /*printf("%lu, %lu, %lu\n", b.size(), vb.size(), tb.size());
        int sum =0;
        for(int i=0;i<data.size();i++)    
           sum=sum+data.at(i).size();
        printf("sum=%d\n",sum);*/
        
        
        printf("\n\n\n");  


	//double lastLH = 1e100;

	//=================Start iteration================================
	while (iter < 100) {
		//Calculate the correct rate
      //		printf("%citer: %d, ", 13, iter);
		//myself
		printf("%d\t",iter);
		
		//*******************************************************************************

		//double LH = check();
		//updateWordsExists();
		//if (iter)
		check();

		iter++;
		/*if(LH > lastLH){
			alpha = 0.0001;
			}
			lastLH = LH;*/


		double lastTime = getTime();
		//memset(gA, 0, sizeof(double)*class_size*H);
		//memset(gB, 0, sizeof(double)*H*input_size);

		for (int i = 0; i < N; i++) {
			swap(order[i], order[rand() % N]);
		}
		//double tlambda = lambda;
		double err0 = 0;
		int cnt = 0;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
		for (int i = 0; i < N; i++) {
			//lambda = 0;
			//if (i % 10 == 0)
			//	lambda = tlambda;
			int s = order[i];
			//data_t *x = data + s*window_size;
			int ans = b[s];

			int tmp, output;
			double terr = checkCase(data[s], ans, tmp, output, s, true);
#pragma omp critical
			{
				cnt++;
				err0 += terr;
				if ((cnt % 10) == 0) {
					//	printf("%cIter: %3d\t   Progress: %.2f%%   Err: %.2lf   Words/sec: %.1f ",
					//		13, iter, 100.*cnt / N, err0 / (cnt + 1), cnt / (getTime() - lastTime));
				}
			}
		}
		//lambda = tlambda;
		//for(int i = 0; i < vN; i++){
		//	int s = i;
		//	data_t *x = vdata + s * window_size;
		//	int ans = vb[s];
		//	int tmp;
		//	checkCase(x, ans, tmp, true);

		//	if ((i%100)==0){
		//	//	printf("%cIter: %3d\t   Progress: %.2f%%   Words/sec: %.1f ", 13, iter, 100.*i/N, i/(getTime()-lastTime));
		//	}
		//}
		//printf("%c", 13);
	}


	//myself
	printf("\n\n\n");
	for(int i=0;i<SROW-TRAINSUM;i++){
		printf("%d ",lastOutput[i]);
	}
	printf("\n\n");
	for(int i=0;i<SROW-TRAINSUM;i++){
                printf("%d ",testLabel[i]);
        }	

	//**********************************************************************************	

	return 0;
}
