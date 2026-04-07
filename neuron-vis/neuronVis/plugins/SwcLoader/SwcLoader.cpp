#include "SwcLoader.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <tuple>
namespace neuronVis {
	void Trim(std::string & str) {
		std::string blanks("\f\v\r\t\n ");
		str.erase(0, str.find_first_not_of(blanks));
		str.erase(str.find_last_not_of(blanks) + 1);
	}


	std::vector<std::string> splitString(std::string srcStr, std::string delimStr, bool repeatedCharIgnored = true)
	{
		std::vector<std::string> resultStringVector;
		std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c) {
			if (delimStr.find(c) != std::string::npos)
			{
				return true;
			}
			else { return false; }
		}/*pred*/, delimStr.at(0)
			);//将出现的所有分隔符都替换成为一个相同的字符（分隔符字符串的第一个）
		size_t pos = srcStr.find(delimStr.at(0));
		std::string addedString = "";
		while (pos != std::string::npos) {
			addedString = srcStr.substr(0, pos);
			if (!addedString.empty() || !repeatedCharIgnored) {
				resultStringVector.push_back(addedString);
			}
			srcStr.erase(srcStr.begin(), srcStr.begin() + pos + 1);
			pos = srcStr.find(delimStr.at(0));
		}
		addedString = srcStr;
		if (!addedString.empty() || !repeatedCharIgnored) {
			resultStringVector.push_back(addedString);
		}
		return resultStringVector;
	}

	NeuronTree::NeuronTree() {

	}

	NeuronTree::~NeuronTree() {

	}
	void NeuronTree::parse() {
		for (int i = 0; i < points.size(); i++) {
			auto point = points[i];
			if (point->parentIndex == -1) {
				root = point;
			}
			for (int j = i; j >= 0; j--) {
				auto p = points[j];
				if (point->parentIndex == p->index) {
					point->parent = p;
					p->children.push_back(point);
					if (p->children.size() == 2)
						branchs.push_back(p);
					break;
				}
			}
		}


		getEdges();

		//getedges

	}
	void NeuronTree::getEdges() {

		edges.push_back({ root });
		for (auto point : branchs) {
			for (int i = 0; i < point->children.size(); i++) {
				edges.push_back({ point });
			}
		}
		//clock_t startime, endtime;
		//startime=clock();//记录开始时间
		for (auto point : points) {
			for (int i = 0; i < edges.size(); i++) {
				auto edge = edges[i];
				auto lastp = edge[edge.size() - 1];
				if (point->parent == lastp && point->children.size() <= 1) {
					edge.push_back(point);
					break;
				}
			}
			if (point->children.size() == 0) {
				terminals.push_back(point);
			}
		}
		//endtime=clock();//记录结束时间
		//double tot_time = (double)(endtime - startime);
		//std::cout<<tot_time/1000<<std::endl;
	}

	SwcLoader::SwcLoader() {

	}
	SwcLoader::~SwcLoader() {

	}

	std::shared_ptr<NeuronTree> SwcLoader::ReadSWC2(std::string fileName) {
		std::ifstream ifs;

		ifs.open(fileName, std::ios::in);

		if (!ifs.is_open()) {

			std::cout << "Open file " << fileName << " error!" << std::endl;

			return nullptr;
		}

		std::string buf;

		// skip comment
		while (getline(ifs, buf)) {
			std::cout << buf << std::endl;
			Trim(buf);
			char c = buf.at(0);
			char sharp = '#';
			if (c != sharp)
				break;
		}
		// start parse

		std::cout << buf << std::endl;
		auto point = splitString(buf, split);
		std::shared_ptr<NeuronTree> neuron = std::make_shared<NeuronTree>();

		auto p = std::make_shared<Point>(point);
		neuron->points.push_back(p);
		while (getline(ifs, buf)) {
			auto point = splitString(buf, split);
			auto p = std::make_shared<Point>(point);
			neuron->points.push_back(p);

		}
		neuron->parse();
		return neuron;
	}
	GraphPriv SwcLoader::ReadSWC(std::string fileName)
	{
		std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> swc{};

		try {

			std::ifstream ifs{ fileName };
			if (!ifs)
				std::cout << "File open failed : " << fileName << std::endl;
			ifs.precision(13);
			ifs.setf(std::ios::scientific);

			std::string line;
			while (std::getline(ifs, line)) {
				if (line.size() == 0) continue;
				if (line[0] == '#') continue;
				std::istringstream iss{ line };
				iss.precision(13);
				iss.setf(std::ios::scientific);

				int64_t id, par;
				int16_t type;
				double x, y, z, r;
				iss >> id >> type >> x >> y >> z >> r >> par;
				if (!iss)
					std::cout << "Failed to parse line: " << line << std::endl;
				swc.push_back(std::make_tuple(id, type, x, y, z, r, par));
			}
			if (!ifs.eof())
				std::cout << "Failed to read file." << std::endl;
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;

		}

		if (swc.size() == 0) {
			std::cerr << "Empty SWC file.\n";
			return GraphPriv();
		}
		GraphPriv graphdata;
		graphdata.fromSwc(swc);
		return graphdata;
	}
}

extern "C" {
    neuronVis::SwcLoader loader;
    void readswc(char * filename) {
        std::string file=filename;
        std::cout<<file<<std::endl;
        loader.ReadSWC(filename);
        return;
      }
     int test(int a) {
         readswc("dtest");
         return a;
      }
    char* strTest(char* pVal){
        std::cout<<pVal<<std::endl;
        return pVal;
    }

}
