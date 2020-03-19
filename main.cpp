//#define TEST
//#define LOCAL
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;



struct Data {
	vector<double> features;
	int label;
	Data(vector<double> f, int l) : features(f), label(l)
	{}
};
struct Param {
	vector<double> wtSet;
};


class LR {
public:
	void train();
	void predict();
	int loadModel();
	int storeModel();
	LR(string trainFile, string testFile, string predictOutFile);

private:
	vector<Data> trainDataSet;
	vector<Data> testDataSet;
	vector<int> predictVec;
	Param param;
	string trainFile;
	string testFile;
	string predictOutFile;
	string weightParamFile = "modelweight.txt";

private:
	bool init();
	bool loadTrainData();
	bool loadTestData();
	int storePredict(vector<int> &predict);
	void initParam();
	double wxbCalc(const Data &data);
	double sigmoidCalc(const double wxb);
	double lossCal();
	double gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec);

private:
	int featuresNum;
	const double wtInitV = 0.0;
	double stepSize = 0.001;
	const int maxIterTimes = 30;
	const double predictTrueThresh = 0.4;
	const double decay = 0.9;
	const int train_data_num = 1000;
	const int decay_num = 10;

};

LR::LR(string trainF, string testF, string predictOutF)
{
	trainFile = trainF;
	testFile = testF;
	predictOutFile = predictOutF;
	featuresNum = 0;
	init();
}

bool LR::loadTrainData()
{
	int count = 0;
	clock_t start, finish;
	double total_time;
	start = clock();
	ifstream infile(trainFile.c_str());
	string line;

	if (!infile) {
		std::cout << "打开训练文件失败" << endl;
		exit(0);
	}
	int count_train = 0;
	while (getline(infile, line)) {
        count_train++;
		if (count_train > train_data_num) { break; }
		
		char *buf;
		vector<double> feature;
		char *s_input = (char *)line.c_str();
		const char * split = ",";
		char *p = strtok_s(s_input, split, &buf);
		double a;
		while (p != NULL) {
			a = atof(p);
			feature.push_back(a);
			p = strtok_s(NULL, split, &buf);
		}
		int ftf;
		ftf = (int)feature.back();
		feature.pop_back();
		trainDataSet.push_back(Data(feature, ftf));
	}


	/*while (infile) {
		
		getline(infile, line);
		if (line.size() > featuresNum) {
			stringstream sin(line);
			char ch;
			double dataV;
			int i;
			vector<double> feature;
			i = 0;
			count++;
			cout << count << endl;

			while (sin) {
				
					sin >> dataV;
					feature.push_back(dataV);
					sin >> ch;
					i++;
				
				
			}
			int ftf;
			ftf = (int)feature.back();
			feature.pop_back();
			trainDataSet.push_back(Data(feature, ftf));
		}
	}
	*/

	infile.close();
	finish = clock();
	total_time = double(finish - start) / 1000;
	cout << " read train data time is " << total_time << "s" << endl;
	return true;
}

void LR::initParam()
{
	int i;
	
	//to do random(-0.5,0.5)
	for (i = 0; i < featuresNum; i++) {
		param.wtSet.push_back(wtInitV);
	}
}

bool LR::init()
{
	trainDataSet.clear();
	bool status = loadTrainData();
	if (status != true) {
		return false;
	}
	featuresNum = trainDataSet[0].features.size();
	param.wtSet.clear();
	initParam();
	return true;
}


double LR::wxbCalc(const Data &data)
{
	double mulSum = 0.0L;
	int i;
	double wtv, feav;
	for (i = 0; i < param.wtSet.size(); i++) {
		wtv = param.wtSet[i];
		feav = data.features[i];
		mulSum += wtv * feav;
	}

	return mulSum;
}

inline double LR::sigmoidCalc(const double wxb)
{
	double expv = exp(-1 * wxb);
	double expvInv = 1 / (1 + expv);
	return expvInv;
}


double LR::lossCal()
{
	double lossV = 0.0L;
	int i;

	for (i = 0; i < trainDataSet.size(); i++) {
		lossV -= trainDataSet[i].label * log(sigmoidCalc(wxbCalc(trainDataSet[i])));
		lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoidCalc(wxbCalc(trainDataSet[i])));
	}
	lossV /= trainDataSet.size();
	return lossV;
}


double LR::gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec)
{
	double gsV = 0.0L;
	int i;
	double sigv, label;
	for (i = 0; i < dataSet.size(); i++) {
		sigv = sigmoidVec[i];
		label = dataSet[i].label;
		gsV += (label - sigv) * (dataSet[i].features[index]);
	}

	gsV = gsV / dataSet.size();
	return gsV;
}

void LR::train()
{
	clock_t begin, end;
	begin = clock();
	double sigmoidVal;
	double wxbVal;
	int i, j, k;
	int count_iter = 0;
	int m = trainDataSet.size();
	int n = param.wtSet.size();
	double alpha;
	double error;
	//int counter = 0;
	for (j = 0; j < maxIterTimes; j++) {
		for (i = 0; i < m; i++) {//line
			//alpha = 4 / (1.0 + j + i) + 0.001;
			//alpha = 0.1;
			wxbVal = wxbCalc(trainDataSet[i]);
			sigmoidVal = sigmoidCalc(wxbVal);
			error = trainDataSet[i].label - sigmoidVal;
			for (k = 0; k < param.wtSet.size(); k++) {
				param.wtSet[k] += stepSize * error * (trainDataSet[i].features)[k];
			}
		}
		if (j % decay_num == 0) {
			stepSize *= decay;
		}

	}



	/*for (i = 0; i < maxIterTimes; i++) {
		count_iter++;
		vector<double> sigmoidVec;

		for (j = 0; j < trainDataSet.size(); j++) {
			wxbVal = wxbCalc(trainDataSet[j]);
			sigmoidVal = sigmoidCalc(wxbVal);
			sigmoidVec.push_back(sigmoidVal);
		}

		for (j = 0; j < param.wtSet.size(); j++) {
			param.wtSet[j] += stepSize * gradientSlope(trainDataSet, j, sigmoidVec);
		}
		if (count_iter % decay_num == 0) {
			stepSize *= decay;
		}
}*/

		/*if (i % train_show_step == 0) {
			cout << "iter " << i << ". updated weight value is : ";
			for (j = 0; j < param.wtSet.size(); j++) {
				cout << param.wtSet[j] << "  ";
			}
			cout << endl;
		}*/
	
	end = clock();
	double total_time = double(end - begin) / 1000;
	cout << "train time is:" << total_time << "s" << endl;
} 

void LR::predict()
{
	clock_t begin, end;
	begin = clock();
	double sigVal;
	int predictVal;

	loadTestData();
	for (int j = 0; j < testDataSet.size(); j++) {
		sigVal = sigmoidCalc(wxbCalc(testDataSet[j]));
		predictVal = sigVal >= predictTrueThresh ? 1 : 0;
		predictVec.push_back(predictVal);
	}

	storePredict(predictVec);
	end = clock();
	double total_time = double(end - begin) / 1000;
	cout << "predict time is:" << total_time << "s" <<endl;
}

int LR::loadModel()
{
	string line;
	int i;
	vector<double> wtTmp;
	double dbt;

	ifstream fin(weightParamFile.c_str());
	if (!fin) {
		cout << "打开模型参数文件失败" << endl;
		exit(0);
	}

	getline(fin, line);
	stringstream sin(line);
	for (i = 0; i < featuresNum; i++) {
		char c = sin.peek();
		if (c == -1) {
			cout << "模型参数数量少于特征数量，退出" << endl;
			return -1;
		}
		sin >> dbt;
		wtTmp.push_back(dbt);
	}
	param.wtSet.swap(wtTmp);
	fin.close();
	return 0;
}

int LR::storeModel()
{
	string line;
	int i;

	ofstream fout(weightParamFile.c_str());
	if (!fout.is_open()) {
		cout << "打开模型参数文件失败" << endl;
	}
	if (param.wtSet.size() < featuresNum) {
		cout << "wtSet size is " << param.wtSet.size() << endl;
	}
	for (i = 0; i < featuresNum; i++) {
		fout << param.wtSet[i] << " ";
	}
	fout.close();
	return 0;
}


bool LR::loadTestData()
{
	clock_t begin, end;
	begin = clock();
	ifstream infile(testFile.c_str());
	string line;

	if (!infile) {
		cout << "打开测试文件失败" << endl;
		exit(0);
	}
	while (getline(infile, line)) {
		char *buf;
		vector<double> feature;
		char *s_input = (char *)line.c_str();
		const char * split = ",";
		char *p = strtok_s(s_input, split, &buf);
		double a;
		while (p != NULL) {
			a = atof(p);
			feature.push_back(a);
			p = strtok_s(NULL, split, &buf);
		}
		testDataSet.push_back(Data(feature, 0));
	}
	infile.close();
	end = clock();
	double total_time = double(end - begin) / 1000;
	cout << "read test data time is:" << total_time << "s" << endl;
	return true;
}

bool loadAnswerData(string awFile, vector<int> &awVec)
{
	ifstream infile(awFile.c_str());
	if (!infile) {
		cout << "打开答案文件失败" << endl;
		exit(0);
	}

	string line;
	double a;
	while (getline(infile, line)) {
		char *s_input = (char *)line.c_str();
		a = atof(s_input);
		awVec.push_back(a);
	}

	

	infile.close();
	return true;
}

int LR::storePredict(vector<int> &predict)
{
	string line;
	int i;

	ofstream fout(predictOutFile.c_str());
	if (!fout.is_open()) {
		cout << "打开预测结果文件失败" << endl;
	}
	for (i = 0; i < predict.size(); i++) {
		fout << predict[i] << endl;
	}
	fout.close();
	return 0;
}

int main(int argc, char *argv[])
{
	vector<int> answerVec;
	vector<int> predictVec;
	int correctCount;
	double accurate;
#ifdef LOCAL
	string trainFile = "D://CODE//huawei//2020_3//data//train_data.txt";
	string testFile = "D://CODE//huawei//2020_3//data//test_data.txt";
	string predictFile = "D://CODE//huawei//2020_3//data//student//result.txt";
	string answerFile = "D://CODE//huawei//2020_3//data//answer.txt";
#else
	string trainFile = "/data/train_data.txt";
	string testFile = "/data/test_data.txt";
	string predictFile = "/projects/student/result.txt";
	string answerFile = "/projects/student/answer.txt";
#endif

	





	LR logist(trainFile, testFile, predictFile);

	cout << "ready to train model" << endl;
	logist.train();

	cout << "training ends, ready to store the model" << endl;
	logist.storeModel();

#ifdef TEST
	cout << "ready to load answer data" << endl;
	loadAnswerData(answerFile, answerVec);
#endif

	cout << "let's have a prediction test" << endl;
	logist.predict();

#ifdef TEST
	loadAnswerData(predictFile, predictVec);
	cout << "test data set size is " << predictVec.size() << endl;
	correctCount = 0;
	for (int j = 0; j < predictVec.size(); j++) {
		if (j < answerVec.size()) {
			if (answerVec[j] == predictVec[j]) {
				correctCount++;
			}
		}
		else {
			cout << "answer size less than the real predicted value" << endl;
		}
	}

	accurate = ((double)correctCount) / answerVec.size();
	cout << "the prediction accuracy is " << accurate << endl;
#endif

	return 0;
}
