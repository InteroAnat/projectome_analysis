#ifndef _FNT_GRAPH_PRIV_H_
#define _FNT_GRAPH_PRIV_H_

#include "graph.h"


struct VertexPriv: VertexData {
	static Vertex alloc() {
		Vertex v;
		v.priv=new VertexPriv{};
		return v;
	}
	static VertexPriv* get(Vertex v) {
		return v.priv;
	}
	static void free(Vertex v) {
		delete v.priv;
		v.priv=nullptr;
	}
};

struct EdgePriv: EdgeData {
	int vaoi;
	static Edge alloc() {
		Edge v;
		v.priv=new EdgePriv{};
		return v;
	}
	static EdgePriv* get(Edge v) {
		return v.priv;
	}
	static void free(Edge v) {
		delete v.priv;
		v.priv=nullptr;
	}
};

struct TreePriv: TreeData {
	bool completed_prev;
	static Tree alloc() {
		Tree v;
		v.priv=new TreePriv{};
		return v;
	}
	static TreePriv* get(Tree v) {
		return v.priv;
	}
	static void free(Tree v) {
		delete v.priv;
		v.priv=nullptr;
	}
};

class  GraphPriv:public GraphData {
public:
	GraphPriv(){}
	~GraphPriv()
	{

	}
	static Graph alloc() {
		Graph g;
		g.priv=new GraphPriv{};
		return g;
	}
	static GraphPriv* get(Graph g) {
		return g.priv;
	}
	static void free(Graph v) {
		delete v.priv;
		v.priv=nullptr;
	}
	void load(std::istream& fs);
	void save(std::ostream& fs) const;
	void updateModel();

	//std::vector<std::string> loadFnt(const char* filename);
	//void saveFnt(const char* filename, const std::vector<std::string>& header) const;

	void fromSwc(const std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>>& swc);
	void fromSwc(const char* filename);
	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> toSwc() const;
};

extern const char* FNT_MAGIC;
extern const char* FNTZ_MAGIC;

#endif
