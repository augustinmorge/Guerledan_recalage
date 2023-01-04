#include <iostream>
#include <fstream>
#include <cassert>

struct Header {
	uint8_t headerId;
	uint8_t dataSourceId;
	uint16_t nbrBytes;
	uint8_t spare;
	uint8_t nbrDataTypes;
};

void read_header(Header header){
	std::cout << std::hex << "header.headerId: " << (int)header.headerId << std::endl;
	std::cout << std::hex << "header.dataSourceId: " << (int)header.dataSourceId << std::endl;
	std::cout << std::dec << "header.nbrBytes: " << header.nbrBytes << std::endl;
	std::cout << std::dec << "header.spare: " << (int)header.spare << std::endl;
	std::cout << std::dec << "header.nbrDataTypes: " << (int)header.nbrDataTypes << std::endl;
}

void load_ping(std::ifstream& datafile){
  // Structure permettant de lire la première ligne :
  uint32_t pos = datafile.tellg();
  Header header;
  //Lit le marker
  datafile.read((char*)&header, sizeof(header));

  if (datafile.eof()) {
		std::cout << "End of the document" << '\n';
    return ;
  }

  //Vérification du premier header
	read_header(header);
  assert(header.headerId==0x7F);

  // for (size_t i = 0; i < header.nbrBytes; i++) {
  for (size_t i = 0; i < (int)header.nbrDataTypes; i++) {
    //Lecture des octets de taille des lignes
    uint16_t octets_starts;
    datafile.read((char*)&octets_starts,sizeof(uint16_t)); //Read 2 octets at the end of each tab which corresponds to the size of the line
    //Lecture de la ligne i du header
    std::cout << octets_starts << '\n';
  }
  //On passe au header suivant
  datafile.seekg(pos + header.nbrBytes + sizeof(header));
}

int main() {
	std::string filename="../dvl1.log";
	std::ifstream datafile;
	datafile.open(filename.c_str(),std::ios::binary | std::ios::in);

	if (!datafile.is_open())
		throw "Unable to load data file";

	// read_header(header);
	while (!datafile.eof()){
		load_ping(datafile);
	}

}
