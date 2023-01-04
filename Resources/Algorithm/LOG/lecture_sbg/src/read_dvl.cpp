#include <iostream>
#include <fstream>

using namespace std;

struct Header {
	uint8_t headerId;
	uint8_t dataSourceId;
	uint16_t nbrBytes;
	uint8_t spare;
	uint8_t nbrDataTypes;
	uint16_t offsetDataType1;
};

int main() {
	string filename="../20221013_13h_guerledan_test_dvl.001";
	ifstream datafile;
	datafile.open(filename.c_str(),ios::binary | ios::in);

	if (!datafile.is_open()) 
		throw "Unable to load data file";
	Header header;
	datafile.read((char*)&header.headerId,sizeof(header.headerId));
	cout << header.headerId << endl;
}