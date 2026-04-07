
#ifndef SWCLOADER_H
#define SWCLOADER_H
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include "graph.priv.h"
namespace neuronVis {
	template <class Type>
	Type stringToNum(const std::string& str)
	{
		std::istringstream iss(str);
		Type num;
		iss >> num;
		return num;
	}

	struct Coord {
		float x, y, z;
	};
	class Point {
	public:
		Point(std::vector<std::string> point) {
			index = stringToNum<int>(point[0]);
			type = stringToNum<int>(point[1]);
			coord.x = stringToNum<float>(point[2]);
			coord.y = stringToNum<float>(point[3]);
			coord.z = stringToNum<float>(point[4]);
			ratio = stringToNum<float>(point[5]);
			parentIndex = stringToNum<int>(point[6]);
		}
	public:
		int parentIndex = -1;
		int type = 0;
		std::vector<std::shared_ptr<Point> > children;
		std::shared_ptr<Point> parent;
		int index = -1;
		Coord coord;
		float ratio = 0.0;

	};
	class Edge {
	public:
		Edge() {}
	private:
	};

	class NeuronTree {
	public:
		NeuronTree();
		~NeuronTree();
		void parse();
		void getEdges();
	public:
		std::shared_ptr<Point> root;
		std::vector<std::vector<std::shared_ptr<Point>>> edges;
		std::vector<std::shared_ptr<Point>> points;
		std::vector<std::shared_ptr<Point>> branchs;
		std::vector<std::shared_ptr<Point>> terminals;
		std::vector<Coord> coords;
	};

	class SwcLoader {
	public:
		SwcLoader();
		std::shared_ptr<NeuronTree> ReadSWC2(std::string fileName);
		GraphPriv ReadSWC(std::string fileName);
		~SwcLoader();
		std::string split = "\f\v\r\t\n ";
	};

}


#endif
