#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

#pragma pack(1)

struct Header {
	int8_t headerId;
	int8_t dataSourceId;
	int16_t nbrBytes;
	int8_t spare;
	int8_t nbrDataTypes;
	int16_t offsetDataType1;
	int16_t offsetDataType2;
	int16_t offsetDataType3;
};

struct FixedData {
	int16_t FixedLeaderId;
	int8_t CPU_VER;
	int8_t CPU_REV;
	int16_t SystemConfig;
	int8_t RealSim_Flag;
	int8_t LagLength;
	int8_t NumberofBeams;
	int8_t NumberofCells;
	int16_t PingsPerEnsemble;
	int16_t DepthCellLength;
	int16_t BlankAfterTransmit;
	int8_t ProfilingMode;
	int8_t LowCorrThresh;
	int8_t CodeReps;
	int8_t Rien;
	int16_t ErrorVelMax;
	int8_t TppMinutes;
	int8_t TppSeconds;
	int8_t TppHundredths;
	int8_t CoordinateTransform;
	int16_t HeadingAlignment;
	int16_t HeadingBias;
	int8_t SensorSource;
	int8_t SensorsAvailable;
	int16_t Bin1Distance;
	int16_t XMITPulseLength;
	int16_t Spare;
	int8_t FalseTargetThresh;
	int8_t Spare2;
	int16_t TransmitLagDistance;
	int64_t Spare3;
	int16_t SystemBandWidth;
	int16_t Spare3_4;
	int32_t SystemSerialNumber;
};

struct VariableData {
	int16_t VariableId;
	int16_t EnsembleNbr;
	int8_t RTC_y_m_d_h_m_s_h[7];
	int8_t Ensemble;
	int16_t BitResult;
	int16_t SpeedofSound;
	int16_t DepthTransducer;
	int16_t Heading;
	int16_t Pitch;
	int16_t Roll;
	int16_t Salinty;
	int16_t Temperature;
	int8_t MPTMin_Sec_Hun_STDHdgPitchRoll[6];
	int8_t ADCChannel[8];
	int32_t ESW; //Error Status Word
	int16_t Spare;
	int32_t Pressure;
	int32_t PressureVar;
	int8_t Spare2[10];
	int8_t LeakStat;
	int16_t LeakACount;
	int16_t LeakBCount;
	int32_t TXVoltage_Current;
	int16_t TransducerImp;
};

struct BT {
	int16_t BTID;
	int16_t BT_pingsPerEnsemble;
	int16_t Reserved;
	int8_t BT_corrMin;
	int8_t BT_evalMin;
	int8_t Reserved2;
	int8_t BTMode;
	int16_t BTErrMax;
	int32_t Reserved3;
	int16_t BT_range[4];
	int16_t BT_vel[4];
	int8_t BT_Corr[4];
	int8_t Eval_AMP[4];
	int8_t BTGood[4];
	int16_t RefLayerMin;
	int16_t RefLayerNear;
	int16_t RefLayerFar;
	int16_t RefLayerVel[4];
	int8_t RefCorr[4];
	int8_t RefInt[4];
	int8_t RefGood[4];
	int16_t MaxDepth;
	int8_t RssiAMP[4];
	int8_t Gain;
	int8_t BT_rangeMSB[4];
};


struct Checksum {
	int16_t Checksum_Data;
};

void readingHeader(ifstream &datafile, Header &header) {
	datafile.read((char*)&header,sizeof(header));
}

void readingFixed(ifstream &datafile, FixedData &fixed) {
	datafile.read((char*)&fixed,sizeof(fixed));
}

void readingVariable(ifstream &datafile, VariableData &var) {
	// var.RTC_y_m_d_h_m_s_h = new int8_t[7];
	// var.MPTMin_Sec_Hun_STDHdgPitchRoll = new int8_t[6];
	// var.ADCChannel  = new int8_t[8];
	// var.Spare2 = new int8_t[10];
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
	string filename="../dvl_final.out";
	string filename2="../dvl_final.txt";

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
			// cout << bt.BT_vel[0] << " | " << bt.BT_vel[1] << " | " << bt.BT_vel[2] << " | " << bt.BT_vel[3] << endl;
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
			cout << j << endl;
			cout << "End of file" << endl;
			bt_data.close();
			break;
		}
	}
	bt_data.close();
}

