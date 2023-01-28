#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

#pragma pack(1)

struct Header {
	uint8_t headerId;
	uint8_t dataSourceId;
	uint16_t nbrBytes;
	uint8_t spare;
	uint8_t nbrDataTypes;
	uint16_t offsetDataType1;
	uint16_t offsetDataType2;
	uint16_t offsetDataType3;
};

struct FixedData {
	uint16_t FixedLeaderId;
	uint8_t CPU_VER;
	uint8_t CPU_REV;
	uint16_t SystemConfig;
	uint8_t RealSim_Flag;
	uint8_t LagLength;
	uint8_t NumberofBeams;
	uint8_t NumberofCells;
	uint16_t PingsPerEnsemble;
	uint16_t DepthCellLength;
	uint16_t BlankAfterTransmit;
	uint8_t ProfilingMode;
	uint8_t LowCorrThresh;
	uint8_t CodeReps;
	uint8_t Rien;
	uint16_t ErrorVelMax;
	uint8_t TppMinutes;
	uint8_t TppSeconds;
	uint8_t TppHundredths;
	uint8_t CoordinateTransform;
	uint16_t HeadingAlignment;
	uint16_t HeadingBias;
	uint8_t SensorSource;
	uint8_t SensorsAvailable;
	uint16_t Bin1Distance;
	uint16_t XMITPulseLength;
	uint16_t Spare;
	uint8_t FalseTargetThresh;
	uint8_t Spare2;
	uint16_t TransmitLagDistance;
	uint64_t Spare3;
	uint16_t SystemBandWidth;
	uint16_t Spare3_4;
	uint32_t SystemSerialNumber;
};

struct VariableData {
	uint16_t VariableId;
	uint16_t EnsembleNbr;
	uint8_t RTC_y_m_d_h_m_s_h[7];
	uint8_t Ensemble;
	uint16_t BitResult;
	uint16_t SpeedofSound;
	uint16_t DepthTransducer;
	uint16_t Heading;
	uint16_t Pitch;
	uint16_t Roll;
	uint16_t Salinty;
	uint16_t Temperature;
	uint8_t MPTMin_Sec_Hun_STDHdgPitchRoll[6];
	uint8_t ADCChannel[8];
	uint32_t ESW; //Error Status Word
	uint16_t Spare;
	uint32_t Pressure;
	uint32_t PressureVar;
	uint8_t Spare2[10];
	uint8_t LeakStat;
	uint16_t LeakACount;
	uint16_t LeakBCount;
	uint32_t TXVoltage_Current;
	uint16_t TransducerImp;
};

struct BT {
	uint16_t BTID;
	uint16_t BT_pingsPerEnsemble;
	uint16_t Reserved;
	uint8_t BT_corrMin;
	uint8_t BT_evalMin;
	uint8_t Reserved2;
	uint8_t BTMode;
	uint16_t BTErrMax;
	uint32_t Reserved3;
	uint16_t BT_range[4];
	uint16_t BT_vel[4];
	uint8_t BT_Corr[4];
	uint8_t Eval_AMP[4];
	uint8_t BTGood[4];
	uint16_t RefLayerMin;
	uint16_t RefLayerNear;
	uint16_t RefLayerFar;
	uint16_t RefLayerVel[4];
	uint8_t RefCorr[4];
	uint8_t RefInt[4];
	uint8_t RefGood[4];
	uint16_t MaxDepth;
	uint8_t RssiAMP[4];
	uint8_t Gain;
	uint8_t BT_rangeMSB[4];
};


struct Checksum {
	uint16_t Checksum_Data;
};

void readingHeader(ifstream &datafile, Header &header) {
	datafile.read((char*)&header,sizeof(header));
}

void readingFixed(ifstream &datafile, FixedData &fixed) {
	datafile.read((char*)&fixed,sizeof(fixed));
}

void readingVariable(ifstream &datafile, VariableData &var) {
	// var.RTC_y_m_d_h_m_s_h = new uint8_t[7];
	// var.MPTMin_Sec_Hun_STDHdgPitchRoll = new uint8_t[6];
	// var.ADCChannel  = new uint8_t[8];
	// var.Spare2 = new uint8_t[10];
	datafile.read((char*)&var,sizeof(var));
}

void readingChecksum(ifstream &datafile, Checksum &checksum) {
	datafile.read((char*)&checksum.Checksum_Data,sizeof(checksum.Checksum_Data));
}

void readingBT (ifstream &datafile, BT &bt) {
	datafile.read((char*)&bt,sizeof(bt));
}

// int readLSBMSBbits(ifstream &datafile, int nbLSB, int nbMSB) {
// 	int result=0;
// 	unsigned char byte;
// 	for (int i = 0; i < nbLSB; ++i)
// 	{

// 	}
// 	for (int i = 0; i < nbMSB; ++i)
// 	{
// 		byte=datafile.get();
// 		result = result << 8 | (int)byte;
// 	}
// 	return result;
// }

int main() {
	string filename="../dvl1.log";
	string filename2="../beuteu.txt";

	ifstream datafile;
	ofstream bt_data;

	datafile.open(filename.c_str(),ios::binary | ios::in);
	bt_data.open(filename2.c_str(),ios::out | ios::in );
	if (!datafile.is_open())
		throw "Unable to load data file";
	Header header;
	FixedData fixed;
	VariableData var;
	BT bt;
	Checksum checksum;
	bt_data << "Year | Month | Day | Hour | Minute | Second | Hundredth | BT Range #1 | #2 | #3 | #4 | BT Vel 1-2 | BT Vel 3-4 | BT Vel Face | BT Vel Std" << endl;
	for (int j = 0; j < 17420; ++j)
	{
		if (!datafile.eof()) {
			readingHeader(datafile,header);
			readingFixed(datafile,fixed);
			readingVariable(datafile,var);
			readingBT(datafile,bt);
			readingChecksum(datafile,checksum);
			// cout << "Pitch : "<< var.Pitch << " | Roll : " << var.Roll << endl;
			cout << bt.BT_range[0] << " | " << bt.BT_range[1] << " | " << bt.BT_range[2] << " | " << bt.BT_range[3] << endl;
			for (int i = 0; i < 7; ++i)
			{
				bt_data << (int) var.RTC_y_m_d_h_m_s_h[i] << ",";
			}
			for (int i = 0; i < 4; ++i)
			{
				bt_data << (int) bt.BT_range[i] << ",";
			}
			for (int i = 0; i < 3; ++i)
			{
				bt_data << (int) bt.BT_vel[i] << ",";
			}
			bt_data << (int) bt.BT_vel[3] << endl;
			// cout << bt.BT_vel[0] << endl;
			// cout << (int) var.RTC_y_m_d_h_m_s_h[5] << endl;
			// for (int i = 0; i < 3; ++i)
			// {
			// 	cout << bt.BT_range[i]/1000. << " | ";
			// }

			// if (bt.BT_range[0]!=0.){ 
			// 	ex_BT=bt.BT_range[0];
			// 	// cout << j << endl;
			// 	bt_data << bt.BT_range[0] << endl;
			// }
			// else {
			// 	bt_data << ex_BT << endl;
			// }

		}
		if (datafile.eof()) {
			cout << "End of file" << endl;
			bt_data.close();
			break;
		}
	}
	bt_data.close();
}




	// Reading First Header
	// uint LSB = intValue & 0x0000FFFF;
	// value = (int16_t)(msb << 8 | lsb);
	// datafile.read((char*)&header.headerId,sizeof(header.headerId));
	// datafile.read((char*)&header.dataSourceId,sizeof(header.dataSourceId));
	// uint8_t lsb;
	// uint8_t msb;
	// datafile.read((char*)&lsb,sizeof(uint8_t));
	// datafile.read((char*)&msb,sizeof(uint8_t));
	// cout << hex << (int)lsb << " , " << (int)msb << endl;
	// cout << msblsb << endl;
	// cout << "Header Id : " << hex << (int) header.headerId << endl;
	// cout << "Data Source Id : "<<(int) header.dataSourceId << endl;
	// cout << "Number of bytes in ensemble : " << dec << header.nbrBytes << endl;
	// cout << "Spare : " << (int) header.spare << endl;
	// cout << "Number of Data Types : " << (int) header.nbrDataTypes << endl;
	// cout << "Offset DataType 1 : " << header.offsetDataType1 << endl;
	// cout << "Offset DataType 2 : " << header.offsetDataType2 << endl;
	// cout << "Offset DataType 3 : " << header.offsetDataType3 << endl;

	// cout << hex << "\nBBT ID : " << bt.BTID << dec << endl;
	// cout << "BT Pings per ensemble : " << bt.BT_pingsPerEnsemble << endl;