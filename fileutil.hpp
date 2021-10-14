#include <string>
#include <map>
#include <stdio.h>
#include <vector>
#include <unordered_map>
using namespace std;

/*
1. Read embedding (the words are in a file) and put them in the hash table.

2. Read in idf, if any (if embedding does not exist, just ignore it)

3. Read in the training set and test set (automatically separate out the validation set?), and put them into a reasonable data structure.
Training set, test set, validation set vec<doc>
doc=vec<word>
word=int -> (string)ch~ch~ch


*/

#define MAX_STRING 10000


map<string, int> dict; //All words in the dictionary str->id
vector<string> vocab; //All words and characters exist here in order. id->str
vector<double> idf;

struct embedding_t {
	int size; //How many variables are contained in it (the number of variables in value) size = element_size * element_num
	int element_size; //The length of a vector
	int element_num; //The number of vectors
	double *value; //All parameters

	void init(int element_size, int element_num) {
		this->element_size = element_size;
		this->element_num = element_num;
		size = element_size * element_num;
		value = new double[size];
	}
};

class WordCh {
public:
	char *word; //
	char *now; //Current word
	char *next; //The end position of the current word
	char erasedCh; //Erased separator (127 in the middle, 0 at the end)
	WordCh(char *w) {
		word = w;
		now = NULL;
		next = NULL;
		erasedCh = 0;
	}

	void findNext() {
		while (*next != 0 && *next != '~') {
			next++;
		}
		erasedCh = *next;
		*next = 0;
	}

	char *NextCh() {
		if (now == NULL) { //The first situation
			now = word;
			next = word;
			findNext();
		} else {
			*next = erasedCh;

			if (erasedCh == 0) { //The end of the situation
				now = NULL;
			} else {
				now = next + 1;
				next++;
				findNext();
			}

		}
		return now;
	}
};


void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;

	while (!feof(fin)) {
		ch = fgetc(fin);

		if (ch == 13) continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}

			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}

		word[a] = ch;
		a++;

		if (a >= MAX_STRING) {
			printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a] = 0;
}

int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;

	map<string, int>::iterator it = dict.find(word);
	if (it == dict.end()) {
		return -1;
	} else {
		return it->second;
	}
}

int ReadEmbedding(embedding_t &words, char *file_name) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("Embedding file not found\n");
		return -1;
	}
	int wordNum, size;
	fscanf(f, "%d", &wordNum);
	fscanf(f, "%d", &size);

	words.init(size, wordNum);

	char str[MAX_STRING];
	for (int b = 0; b < wordNum; b++) {
		char ch;
		fscanf(f, "%s%c", str, &ch);
		string s = str;
		dict[s] = vocab.size();
		vocab.push_back(s);
		for (int a = 0; a < size; a++)
			fscanf(f, "%lf", &words.value[a + b * size]);
		/*
		double len = 0;
		for (int a = 0; a < size; a++)
		len += M[a + b * size] * M[a + b * size];
		len = sqrt(len);
		for (int a = 0; a < size; a++)
		M[a + b * size] /= len;
		*/
	}
	fclose(f);
	return 0;
}


int ReadIDF(char *file_name) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("DF file not found\n");
		return -1;
	}
	vector<int> df(vocab.size()); //id->df (df=0 means that there is no embedding, special judgment is required)

	char str[MAX_STRING];
	int val;
	while (fscanf(f, "%s%d", str, &val) != EOF) {
		string s = str;
		map<string, int>::iterator it = dict.find(s);
		if (it != dict.end()) {
			df[it->second] = val;
		}
	}
	fclose(f);

	double sum = 0;
	for (size_t i = 0; i < df.size(); i++) {
		sum += df[i];
	}
	printf("total df: %.0f\n", sum);
	for (size_t i = 0; i < df.size(); i++) {
		if (df[i] == 0) {
			idf.push_back(0);
		} else {
			idf.push_back(log(sum / df[i]));
		}
	}

	return 0;
}

/*
Read the document, which may be the training set or the test set.
One document per line, where the first number is category information. The original text follows. Each word is expressed in the form of word~word.
*/
void ReadDocs(const char *file_name, vector<doc> &docs, vector<int> &labels, const char *dataset) {
	FILE *fi = fopen(file_name, "rb");
	char label[MAX_STRING];
	int unknown = 0;
	int total = 0;
	while (1) {
		if (feof(fi)) break;
		doc d;
		//The first one is the label, read the label first
		ReadWord(label, fi);
		if (feof(fi)) break;
		int lb = atoi(label);
		labels.push_back(lb);

		while (1) { //Start reading the full text message
			int word = ReadWordIndex(fi);
			if (feof(fi)) break;
			if (word == -1) {
				unknown++; //Low frequency words
				continue;
			}
			//word_count++;
			if (word == 0) break;
			d.push_back(word);
		}
		if (d.size() > 0) {
			docs.push_back(d);
			total += d.size();
		}
	}
	printf("%s data: N(docs):%d, words:%d, unknown(ignore):%d\n", dataset, (int)docs.size(), total, unknown);
}

//==========================================================

typedef vector<string> doc_s;
typedef unordered_map<string, int> umap_si;
/*
Read the document, which may be the training set or the test set.
One document per line, where the first number is category information. The original text follows. Each word is expressed in the form of word~word.
*/
void ReadDocs(const char *file_name, vector<doc_s> &docs, vector<int> &labels, const char *dataset) {
	FILE *fi = fopen(file_name, "rb");
	char label[MAX_STRING];
	int total = 0;
	while (1) {
		if (feof(fi)) break;
		doc_s d;
		//The first one is the label, read the label first
		ReadWord(label, fi);
		if (feof(fi)) break;
		int lb = atoi(label);
		labels.push_back(lb);

		while (1) { //Start reading the full text message
			char word[MAX_STRING];
			ReadWord(word, fi);
			if (feof(fi)) break;

			if (strcmp(word, "</s>") == 0) break;
			d.push_back(word);
		}
		if (d.size() > 0) {
			docs.push_back(d);
			total += d.size();
		}
	}
	fclose(fi);
	printf("%s data: N(docs):%d, words:%d\n", dataset, (int)docs.size(), total);
}

//stat word count in some dataset
umap_si CountWordFreq(vector<doc_s> docs) {
	umap_si freq;
	for (size_t i = 0; i < docs.size(); i++) {
		for (size_t j = 0; j < docs[i].size(); j++) {
			freq[docs[i][j]]++;
		}
	}
	return freq;
}


umap_si ReadEmbeddingWords(const char *file_name, umap_si &id, int &size, embedding_t *words, int withUnknown) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("Embedding file not found\n");
		return umap_si();
	}

	int len = strlen(file_name);
	bool binary = false;
	if (strcmp(".bin", file_name + len - 4) == 0)
		binary = true;

	int wordNum;
	fscanf(f, "%d", &wordNum);
	fscanf(f, "%d", &size);

	FILE *fout = fopen("tmpemb.bin", "wb");

	umap_si ret;
	char str[MAX_STRING];
	int cnt = 1 + withUnknown; //0 padding; 1 unknown
	for (int b = 0; b < wordNum; b++) {
		char ch;
		fscanf(f, "%s%c", str, &ch);
		string s = str;
		int this_id = 0;
		if (id.count(s)) {
			ret[s] = 1;
			id[s] = cnt;
			this_id = cnt;
			cnt++;

			fprintf(fout, "%s ", str);
		}

		double v;
		for (int a = 0; a < size; a++) {
			float ff;
			if (binary) {
				fread(&ff, sizeof(float), 1, f);
				v = ff;
			} else {
				fscanf(f, "%lf", &v);
			}
			if (this_id && words) {
				words->value[a + this_id * size] = v;
				fwrite(&ff, sizeof(float), 1, fout);
			}
		}
		if (this_id && words)
			fprintf(fout, "\n");
	}
	fclose(fout);
	fclose(f);

	//fout = fopen("tmpembhead.bin", "wb");
	//fprintf(fout, "%d %d\n", cnt - 2, size);
	//fclose(fout);
	return ret;
}

umap_si MergeMap(umap_si a, umap_si &b) {
	for (auto i = b.begin(); i != b.end(); i++) {
		a[i->first] += i->second;
	}
	return a;
}

//dict_type: 0 +train; 1 +embedding; 2 +both;
umap_si CreateDict(umap_si train_words, umap_si embedding_words, int dict_type) {
	int threshold = 3; //words exist less than *threshold* times in train will may be consider as UNKNOWN.

	umap_si ret;

	if (dict_type == 0 || dict_type == 2) {
		//add words in train.
		for (auto i = train_words.begin(); i != train_words.end(); i++) {
			if (i->second >= threshold) {
				ret[i->first] = -1;
			}
		}
	}
	if (dict_type == 1 || dict_type == 2) {
		//add words in embedding
		for (auto i = embedding_words.begin(); i != embedding_words.end(); i++) {
			ret[i->first] = -1;
		}
	}

	if (dict_type < 0 || dict_type > 2) {
		printf("unknown dict_type (%d)\n", dict_type);
	}
	return ret;
}

double nextDouble(double s) {
	return rand() / (RAND_MAX + 1.0) * s - s / 2;
}

void InitNonPreTrain(embedding_t words, umap_si &dict, int withUnknown) {
	//0 padding; 1 unknown; ... embedding file ...; ... non pre-train ...;
	int pre = 0;
	umap_si untrain;
	for (auto i = dict.begin(); i != dict.end(); i++) {
		if (i->second != -1)
			pre++;
		else
			untrain[i->first] = 1;
	}
	int offset = 1 + withUnknown;

	int tid = pre + offset;
	for (auto i = untrain.begin(); i != untrain.end(); i++) {
		dict[i->first] = tid;
		tid++;
	}

	double sum = 0, var = 0, var2 = 0;
	for (int i = offset; i < pre + offset; i++) {
		for (int j = 0; j < words.element_size; j++) {
			double v = words.value[words.element_size*i + j];
			sum += v;
			var += v*v;
		}
	}

	int cnt = pre * words.element_size;
	double s = sqrt(var / cnt * 12);
	for (int i = 0; i < words.element_num; i++) {
		if (i >= offset && i < pre + offset)
			continue;
		for (int j = 0; j < words.element_size; j++) {
			double v = nextDouble(s);
			words.value[words.element_size*i + j] = v;
			var2 += v*v;
		}
	}

	int cnt2 = (dict.size() - pre + offset) * words.element_size;

	printf("dict: total:%d, pre-trained:%d, avg:%lf, stddev:%lf, stddev_rand:%lf\n", dict.size(), pre,
		sum / cnt, sqrt(var / (cnt - 1)), sqrt(var2 / (cnt2 - 1)));
}

//map each word from *string* to *int*
vector<doc> SimplifyDocs(vector<doc_s> &docs, umap_si &dict, const char *dataset, int withUnknown) {
	vector<doc> ret;
	int total = 0, unknown = 0;
	for (size_t i = 0; i < docs.size(); i++) {
		doc d;
		doc_s &now = docs[i];
		for (size_t j = 0; j < now.size(); j++) {
			auto it = dict.find(now[j]);
			if (it != dict.end()) {
				d.push_back(it->second);
			} else {
				if (withUnknown)
					d.push_back(1); //unknown
				unknown++;
			}
			total++;
		}
		ret.push_back(d);
	}
	printf("%s data: N(docs):%d, words:%d, unknown:%d\n", dataset, (int)docs.size(), total, unknown);
	return ret;
}


//Do not do cross-validation first, add it if necessary
//Read in train/dev/test, determine the total vocabulary, and read the necessary embedding
//1. First read in all data sets in the form of text
//2. Processing of validation set
//3. Statistics of word count
//4. Generate vocabulary according to requirements
//5. Read embedding
//Parameters: training set file, test set file, verification set ratio (-1 means there is a verification set file, otherwise 0~100), verification set file, word vector file
//Parameter: dict_type: 0 +train; 1 +embedding; 2 +both;
void ReadAllFiles(const char *train_file, const char *test_file, int dev, const char *dev_file,
	const char *file_embedding, int dict_type, embedding_t &words,
	vector<doc> &train_data, vector<int> &train_label,
	vector<doc> &dev_data, vector<int> &dev_label,
	vector<doc> &test_data, vector<int> &test_label, int withUnknown = 1) {

	vector<doc_s> train_doc, dev_doc, test_doc;

	ReadDocs(train_file, train_doc, train_label, "Train");
	ReadDocs(test_file, test_doc, test_label, "Test");
	if (dev == -1) { //read dev file
		ReadDocs(dev_file, dev_doc, dev_label, "Dev");
	} else { //split dev from train
		int N = train_doc.size();
		for (int i = 0; i < N; i++) { //shuffle
			int p = rand() % N;
			swap(train_doc[i], train_doc[p]);
			swap(train_label[i], train_label[p]);
		}

		//add to dev
		int devStart = N * dev / 100;
		for (int i = devStart; i < N; i++) {
			dev_doc.push_back(train_doc[i]);
			dev_label.push_back(train_label[i]);
		}

		//delete from train
		train_doc.erase(train_doc.begin() + devStart, train_doc.end());
		train_label.erase(train_label.begin() + devStart, train_label.end());
	}

	auto train_words = CountWordFreq(train_doc);
	auto dev_words = CountWordFreq(dev_doc);
	auto test_words = CountWordFreq(test_doc);
	auto all_words = MergeMap(MergeMap(train_words, dev_words), test_words);

	int vector_size;

	//only keep useful words in embedding (exist in train/dev/test set)
	printf("read embedding words\n");
	auto embedding_words = ReadEmbeddingWords(file_embedding, all_words, vector_size, NULL, withUnknown);

	//create dictionary according to *dict_type*
	auto dict = CreateDict(train_words, embedding_words, dict_type);

	words.init(vector_size, dict.size() + 1 + withUnknown);
	printf("embedding size:%d, words:%d\n", words.element_size, words.element_num);

	printf("read embedding\n");
	ReadEmbeddingWords(file_embedding, dict, vector_size, &words, withUnknown);

	printf("init embedding not pre-trained\n");
	InitNonPreTrain(words, dict, withUnknown);

        //============================
	//printf("words element num:%d,words element length:%d,words size:%d\n",words.element_num, words.element_size, words.size);

	train_data = SimplifyDocs(train_doc, dict, "Train", withUnknown);
	dev_data = SimplifyDocs(dev_doc, dict, "Dev", withUnknown);
	test_data = SimplifyDocs(test_doc, dict, "Test", withUnknown);
}

