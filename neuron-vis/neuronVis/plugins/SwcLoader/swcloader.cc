
#include <vector>
#include <utility>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <string>
#include <iostream>
#include <stack>
#include <vector>
#include <sstream>
class StreamBuf: public std::streambuf {
	template<typename T> friend class FilterInput;
	template<typename T> friend class FilterOutput;
	public:
	virtual bool close() { return true; }
};

class InputStream: public std::istream {
	std::stack<StreamBuf*> bufs;
	protected:
	public:
	InputStream(): std::istream(std::cin.rdbuf()), bufs{} { }
	InputStream(const char* fn): std::istream(std::cin.rdbuf()), bufs{} { this->open(fn); }
	InputStream(const std::string& fn): std::istream(std::cin.rdbuf()), bufs{} { this->open(fn); }
	InputStream(FILE* f): std::istream(std::cin.rdbuf()), bufs{} { this->open(f); }
	InputStream(const std::vector<char*>* bufs, size_t bufjunk): std::istream(std::cin.rdbuf()), bufs{} { this->open(bufs, bufjunk); }
	InputStream(const InputStream&) =delete;
	~InputStream() {
		while(!bufs.empty()) {
			auto v=bufs.top();
			delete v;
			bufs.pop();
		}
	}

	InputStream& operator=(const InputStream&) =delete;
	
	StreamBuf* rdbuf() const { return bufs.empty()?nullptr:bufs.top(); }

	bool is_open() const { return !bufs.empty(); }
	bool open(const char* fn);
	bool open(const std::string& fn) { return this->open(fn.c_str()); }
	bool open(FILE* f);
	bool open(const std::vector<char*>* bufs, size_t bufjunk);
	bool close() {
		if(bufs.size()==0) {
			return true;
		}
		if(bufs.size()>1) {
			this->setstate(ios_base::failbit);
			return false;
		}
		auto p=bufs.top();
		if(!p->close()) {
			this->setstate(ios_base::failbit);
			return false;
		}
		bufs.pop();
		delete p;
		this->clear();
		return true;
	}

	bool pushGzip();
	bool pop() {
		if(bufs.size()<=1) {
			fprintf(stderr, "already popped\n");
			return true;
		}
		auto p=bufs.top();
		if(!p->close()) {
			fprintf(stderr, "already close failed\n");
			this->setstate(ios_base::failbit);
			return false;
		}
		bufs.pop();
		delete p;
		this->clear();
		return true;
	}
};

class OutputStream: public std::ostream {
	std::stack<StreamBuf*> bufs;
	protected:
	public:
	OutputStream(): std::ostream(std::cout.rdbuf()), bufs{} { }
	OutputStream(const char* fn): std::ostream(std::cout.rdbuf()), bufs{} { this->open(fn); }
	OutputStream(const std::string& fn): std::ostream(std::cout.rdbuf()), bufs{} { this->open(fn); }
	OutputStream(FILE* f): std::ostream(std::cout.rdbuf()), bufs{} { this->open(f); }
	OutputStream(const OutputStream&) =delete;
	~OutputStream() {
		while(!bufs.empty()) {
			auto v=bufs.top();
			delete v;
			bufs.pop();
		}
	}

	OutputStream& operator=(const OutputStream&) =delete;

	StreamBuf* rdbuf() const { return bufs.empty()?nullptr:bufs.top(); }

	bool is_open() const { return !bufs.empty(); }
	bool open(const char* fn);
	bool open(const std::string& fn) { return this->open(fn.c_str()); }
	bool open(FILE* f);
	bool close() {
		if(bufs.size()==0) {
			return true;
		}
		if(bufs.size()>1) {
			this->setstate(ios_base::failbit);
			return false;
		}
		auto p=bufs.top();
		if(!p->close()) {
			this->setstate(ios_base::failbit);
			return false;
		}
		bufs.pop();
		delete p;
		this->clear();
		return true;
	}

	bool pushGzip();
	bool pop() {
		if(bufs.size()<=1) {
			fprintf(stderr, "already popped\n");
			return true;
		}
		auto p=bufs.top();
		if(!p->close()) {
			fprintf(stderr, "already failed tocl\n");
			this->setstate(ios_base::failbit);
			return false;
		}
		bufs.pop();
		delete p;
		this->clear();
		return true;
	}
};
/* A point in the traced neurite. */
struct Point {
	int32_t _x, _y, _z; // position, fixed-point, precision iii.f, in um.
	uint16_t _r; // radius, fixed-point, precision i.f, in um.
	int16_t m; // mark: 0, no mark; >0, different point types; <0, invalid.

	explicit Point(): _x{0}, _y{0}, _z{0}, _r{0}, m{-1} { }
	Point(double x, double y, double z):
		_x{static_cast<int32_t>(lrint(x*256))},
		_y{static_cast<int32_t>(lrint(y*256))},
		_z{static_cast<int32_t>(lrint(z*256))}, _r{0}, m{0} { }
	Point(double x, double y, double z, double r):
		_x{static_cast<int32_t>(lrint(x*256))},
		_y{static_cast<int32_t>(lrint(y*256))},
		_z{static_cast<int32_t>(lrint(z*256))},
		_r{static_cast<uint16_t>(lrint(r*256))}, m{0} { }
	Point(double x, double y, double z, double r, int16_t mark):
		_x{static_cast<int32_t>(lrint(x*256))},
		_y{static_cast<int32_t>(lrint(y*256))},
		_z{static_cast<int32_t>(lrint(z*256))},
		_r{static_cast<uint16_t>(lrint(r*256))}, m{mark} { }

	bool valid() const { return m>=0; }
	double distTo(const Point& b) const {
		double dx=(static_cast<int64_t>(b._x)-_x)/256.0;
		double dy=(static_cast<int64_t>(b._y)-_y)/256.0;
		double dz=(static_cast<int64_t>(b._z)-_z)/256.0;
		return std::sqrt(dx*dx+dy*dy+dz*dz);
	}
	bool operator==(const Point& p) const {
		return _x==p._x && _y==p._y && _z==p._z;
	}

	double x() const { return _x/256.0; }
	double y() const { return _y/256.0; }
	double z() const { return _z/256.0; }
	double r() const { return _r/256.0; }

	void x(double x) { _x=lrint(x*256); }
	void y(double y) { _y=lrint(y*256); }
	void z(double z) { _z=lrint(z*256); }
	void r(double r) { _r=lrint(r*256); }
};

class Vertex;
struct VertexPriv;
class Edge;
struct EdgePriv;
class Tree;
struct TreePriv;

class Vertex {
	friend struct VertexPriv;
	VertexPriv* priv;
	public:
	explicit Vertex(): priv{nullptr} { }
	~Vertex() { }
	Vertex(const Vertex& v): priv{v.priv} { }
	Vertex& operator=(const Vertex& v) { priv=v.priv; return *this; }
	bool operator==(const Vertex& v) const { return v.priv==priv; }
	bool operator!=(const Vertex& v) const { return v.priv!=priv; }

	explicit operator bool() const { return priv; }
	bool operator!() const { return !priv; }

	size_t index() const;
	const std::vector<std::pair<Edge, bool>>& neighbors() const;
	bool finished() const;
	Tree tree() const;
	bool inLoop() const;
	Vertex parentVertex() const;
	Edge parentEdge() const;
	const Point& point() const;
};

class Edge {
	friend struct EdgePriv;
	EdgePriv* priv;
	public:
	explicit Edge(): priv{nullptr} { }
	~Edge() { }
	Edge(const Edge& e): priv{e.priv} { }
	Edge& operator=(const Edge& e) { priv=e.priv; return *this; }
	bool operator==(const Edge& v) const { return v.priv==priv; }
	bool operator!=(const Edge& v) const { return v.priv!=priv; }

	explicit operator bool() const { return priv; }
	bool operator!() const { return !priv; }

	int16_t type() const;
	const std::vector<Point>& points() const;
	size_t length() const;
	Vertex leftVertex() const;
	Vertex rightVertex() const;
	size_t index() const;
	Tree tree() const;
	bool inLoop() const;
	Vertex parentVertex() const;
	Edge parentEdge() const;
};

struct Path {
	std::vector<Point> points;
	Edge edge0;
	size_t index0;
	Edge edge1;
	size_t index1;
	Path(): points{}, edge0{}, edge1{} { }
};

class Tree {
	friend struct TreePriv;
	TreePriv* priv;
	public:
	explicit Tree(): priv{nullptr} { }
	~Tree() { }
	Tree(const Tree& n): priv{n.priv} { }
	Tree& operator=(const Tree& n) { priv=n.priv; return *this; }
	bool operator==(const Tree& v) const { return v.priv==priv; }
	bool operator!=(const Tree& v) const { return v.priv!=priv; }

	explicit operator bool() const { return priv; }
	bool operator!() const { return !priv; }

	const std::string& name() const;
	Vertex root() const;
	bool completed() const;
	size_t index() const;
	bool selected() const;
};

struct VertexData {
	size_t index;
	std::vector<std::pair<Edge, bool>> neighbors; // *
	bool finished; // *
	Tree tree;
	bool inLoop;
	Vertex parentVertex;
	Edge parentEdge;
};

inline size_t Vertex::index() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->index;
}
inline const std::vector<std::pair<Edge, bool>>& Vertex::neighbors() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->neighbors;
}
inline bool Vertex::finished() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->finished;
}
inline Tree Vertex::tree() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->tree;
}
inline bool Vertex::inLoop() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->inLoop;
}
inline Vertex Vertex::parentVertex() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->parentVertex;
}
inline Edge Vertex::parentEdge() const {
	auto data=reinterpret_cast<VertexData*>(priv);
	return data->parentEdge;
}
inline const Point& Vertex::point() const {
	auto& pair=neighbors()[0];
	return pair.first.points()[pair.second?pair.first.length()-1:0];
}

struct EdgeData {
	int16_t type; // *
	std::vector<Point> points; // *
	Vertex leftVertex; // *
	Vertex rightVertex; // *
	size_t index;
	Tree tree;
	Vertex parentVertex;
	Edge parentEdge;
	bool inLoop;
};

inline int16_t Edge::type() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->type;
}
inline const std::vector<Point>& Edge::points() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->points;
}
inline size_t Edge::length() const {
	return points().size();
}
inline Vertex Edge::leftVertex() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->leftVertex;
}
inline Vertex Edge::rightVertex() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->rightVertex;
}
inline size_t Edge::index() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->index;
}
inline Tree Edge::tree() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->tree;
}
inline bool Edge::inLoop() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->inLoop;
}
inline Vertex Edge::parentVertex() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->parentVertex;
}
inline Edge Edge::parentEdge() const {
	auto data=reinterpret_cast<EdgeData*>(priv);
	return data->parentEdge;
}

struct TreeData {
	std::string name; // *
	Vertex root; // *
	bool completed;
	size_t index;
	bool selected;
};

inline const std::string& Tree::name() const {
	auto data=reinterpret_cast<TreeData*>(priv);
	return data->name;
}
inline Vertex Tree::root() const {
	auto data=reinterpret_cast<TreeData*>(priv);
	return data->root;
}
inline bool Tree::completed() const {
	auto data=reinterpret_cast<TreeData*>(priv);
	return data->completed;
}
inline size_t Tree::index() const {
	auto data=reinterpret_cast<TreeData*>(priv);
	return data->index;
}
inline bool Tree::selected() const {
	auto data=reinterpret_cast<TreeData*>(priv);
	return data->selected;
}

struct GraphPriv;
class Graph {
	friend struct GraphPriv;
	GraphPriv* priv;
	public:
	explicit Graph(): priv{nullptr} { }
	~Graph() { }
	Graph(const Graph& g): priv{g.priv} { }
	Graph& operator=(const Graph& g) { priv=g.priv; return *this; }

	explicit operator bool() const { return priv; }
	bool operator!() const { return !priv; }

	const std::vector<Tree>& trees() const;
	const std::vector<Edge>& edges() const;
	const std::vector<Vertex>& vertices() const;
};

struct GraphData {
	std::vector<Edge> edges;
	std::vector<Vertex> vertices;
	std::vector<Tree> trees;
};

inline const std::vector<Tree>& Graph::trees() const {
	auto data=reinterpret_cast<GraphData*>(priv);
	return data->trees;
}
inline const std::vector<Edge>& Graph::edges() const {
	auto data=reinterpret_cast<GraphData*>(priv);
	return data->edges;
}
inline const std::vector<Vertex>& Graph::vertices() const {
	auto data=reinterpret_cast<GraphData*>(priv);
	return data->vertices;
}

struct Position {
	Point point;
	Edge edge;
	size_t index;
	Position(): point{}, edge{}, index{SIZE_MAX} { }
	Position(Edge e, size_t i, const Point& p): point{p}, edge{e}, index{i} { }
};


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

//class  GraphPriv:public GraphData {
//public:
//	GraphPriv(){}
//	~GraphPriv()
//	{
//
//	}
//	static Graph alloc() {
//		Graph g;
//		g.priv=new GraphPriv{};
//		return g;
//	}
//	static GraphPriv* get(Graph g) {
//		return g.priv;
//	}
//	static void free(Graph v) {
//		delete v.priv;
//		v.priv=nullptr;
//	}
//	void load(std::istream& fs);
//	void save(std::ostream& fs) const;
//	void updateModel();
//
//	std::vector<std::string> loadFnt(const char* filename);
//	void saveFnt(const char* filename, const std::vector<std::string>& header) const;
//
//	void fromSwc(const std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>>& swc);
//	void fromSwc(const char* filename);
//	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> toSwc() const;
//};
//
//#include <queue>
//#include <tuple>
//#include <unordered_map>
//#include <fstream>
//#include <array>
//
//void GraphPriv::updateModel() {
//	for(size_t i=0; i<vertices.size(); i++) {
//		auto vp=VertexPriv::get(vertices[i]);
//		vp->index=i;
//		vp->tree=Tree{};
//		vp->inLoop=false;
//		vp->parentVertex=Vertex{};
//		vp->parentEdge=Edge{};
//	}
//	for(size_t i=0; i<edges.size(); i++) {
//		auto ep=EdgePriv::get(edges[i]);
//		ep->index=i;
//		ep->tree=Tree{};
//		ep->inLoop=false;
//		ep->parentVertex=Vertex{};
//		ep->parentEdge=Edge{};
//	}
//	std::queue<Vertex> que{};
//	for(size_t i=0; i<trees.size(); i++) {
//		auto np=TreePriv::get(trees[i]);
//		np->index=i;
//		np->completed_prev=np->completed;
//		np->completed=np->root.finished();
//		auto vp=VertexPriv::get(np->root);
//		vp->tree=trees[i];
//		que.push(np->root);
//	}
//	while(!que.empty()) {
//		auto v=que.front();
//		auto vp=VertexPriv::get(v);
//		que.pop();
//		auto vpnp=TreePriv::get(vp->tree);
//		for(auto& p: vp->neighbors) {
//			auto ep=EdgePriv::get(p.first);
//			if(ep->tree)
//				continue;
//			/*
//			if(vp->parentEdge)
//				ep->type=vp->parentEdge.type();
//				*/
//			auto nv=ep->leftVertex==v ? ep->rightVertex : ep->leftVertex;
//			auto nvp=VertexPriv::get(nv);
//			ep->tree=vp->tree;
//			ep->parentEdge=vp->parentEdge;
//			ep->parentVertex=v;
//			if(!nvp->finished)
//				vpnp->completed=false;
//			if(!nvp->tree) {
//				nvp->tree=vp->tree;
//				nvp->parentVertex=v;
//				nvp->parentEdge=p.first;
//				que.push(nv);
//				continue;
//			}
//
//			// loop found
//			auto nvnp=TreePriv::get(nvp->tree);
//			vpnp->completed=false;
//			nvnp->completed=false;
//			nvp->inLoop=true;
//			ep->inLoop=true;
//			{
//				auto nep=EdgePriv::get(nv.parentEdge());
//				if(nep)
//					nep->inLoop=true;
//			}
//			std::vector<Vertex> pars{};
//			for(auto p=nvp->parentVertex; p; p=p.parentVertex()) {
//				pars.emplace_back(p);
//			}
//			Vertex commp{};
//			for(auto p=v; p; p=p.parentVertex()) {
//				for(auto pp: pars) {
//					if(pp==p) {
//						commp=p;
//						break;
//					}
//				}
//				if(commp)
//					break;
//			}
//			for(auto pp: pars) {
//				auto ppvp=VertexPriv::get(pp);
//				ppvp->inLoop=true;
//				if(pp==commp)
//					break;
//				{
//					auto nep=EdgePriv::get(pp.parentEdge());
//					if(nep)
//						nep->inLoop=true;
//				}
//			}
//			for(auto p=v; p; p=p.parentVertex()) {
//				auto pvp=VertexPriv::get(p);
//				pvp->inLoop=true;
//				if(p==commp)
//					break;
//				{
//					auto nep=EdgePriv::get(p.parentEdge());
//					if(nep)
//						nep->inLoop=true;
//				}
//			}
//		}
//	}
//}
//
//std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> GraphPriv::toSwc() const {
//	std::vector<int64_t> v2nid(vertices.size(), -1);
//	std::vector<bool> edgemask(edges.size(), false);
//	std::queue<Vertex> que{};
//	int64_t nid=1;
//	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> swc;
//	while(true) {
//		if(que.empty()) {
//			for(auto n: trees) {
//				auto s=n.root();
//				if(v2nid[s.index()]==-1) {
//					auto& pos=s.point();
//					swc.push_back(std::make_tuple(nid, 1, pos.x(), pos.y(), pos.z(), pos.r(), -1));
//					v2nid[s.index()]=nid++;
//					que.push(s);
//				}
//			}
//		}
//		if(que.empty()) {
//			for(auto v: vertices) {
//				if(v2nid[v.index()]==-1) {
//					auto& pos=v.point();
//					swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:v.neighbors()[0].first.type(), pos.x(), pos.y(), pos.z(), pos.r(), -1));
//					v2nid[v.index()]=nid++;
//					que.push(v);
//					break;
//				}
//			}
//		}
//		if(que.empty())
//			break;
//
//		while(!que.empty()) {
//			auto v=que.front();
//			que.pop();
//			auto nid_par=v2nid[v.index()];
//			for(auto& ep: v.neighbors()) {
//				auto e=ep.first;
//				if(edgemask[e.index()])
//					continue;
//				if(e.length()==2 && e.points()[0]==e.points()[1]) {
//					Vertex v1;
//					if(ep.second) {
//						v1=e.leftVertex();
//					} else {
//						v1=e.rightVertex();
//					}
//					if(v1.neighbors().size()==1) {
//						edgemask[e.index()]=true;
//						if(v2nid[v1.index()]==-1) {
//							v2nid[v1.index()]=nid++;
//						}
//						continue;
//					}
//				}
//				Vertex v1;
//				auto pnid=nid_par;
//				if(ep.second) {
//					for(size_t j=e.length()-1; j-->0;) {
//						auto& pos=e.points()[j];
//						swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:e.type(), pos.x(), pos.y(), pos.z(), pos.r(), pnid));
//						pnid=nid++;
//					}
//					v1=e.leftVertex();
//				} else {
//					for(size_t j=1; j<e.length(); j++) {
//						auto& pos=e.points()[j];
//						swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:e.type(), pos.x(), pos.y(), pos.z(), pos.r(), pnid));
//						pnid=nid++;
//					}
//					v1=e.rightVertex();
//				}
//				edgemask[e.index()]=true;
//				if(v2nid[v1.index()]==-1) {
//					v2nid[v1.index()]=pnid;
//					que.push(v1);
//				}
//			}
//		}
//	}
//	return swc;
//}
//
//void GraphPriv::fromSwc(const std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>>& swc) {
//	auto N=swc.size();
//	std::vector<int> degree(N, 0);
//	std::unordered_map<int64_t, size_t> id2idx;
//
//	for(size_t idx=0; idx<N; idx++) {
//		auto id=std::get<0>(swc[idx]);
//		auto par=std::get<6>(swc[idx]);
//		auto i=id2idx.find(id);
//		if(i!=id2idx.end())
//			std::runtime_error("Duplicated nodes.");
//		id2idx[id]=idx;
//		if(par!=-1) {
//			auto j=id2idx.find(par);
//			if(j==id2idx.end())
//				std::runtime_error("Node refered but not defined yet.");
//			degree[idx]++;
//			degree[j->second]++;
//		}
//	}
//
//	int nrni=1;
//	std::unordered_map<int64_t, Vertex> id2vert;
//	for(size_t idx=0; idx<N; idx++) {
//		auto id=std::get<0>(swc[idx]);
//		auto par=std::get<6>(swc[idx]);
//		bool node=false;
//		bool soma=false;
//		if(par==-1) {
//			node=true;
//			if(std::get<1>(swc[idx])==1) {
//				soma=true;
//			}
//		} else {
//			switch(degree[idx]) {
//				case 2:
//					break;
//				default:
//					node=true;
//					break;
//			}
//		}
//		if(node) {
//			auto v=VertexPriv::alloc();
//			auto vp=VertexPriv::get(v);
//			vp->finished=true;
//			id2vert[id]=v;
//			vertices.push_back(v);
//			if(soma) {
//				auto n=TreePriv::alloc();
//				auto np=TreePriv::get(n);
//				np->root=v;
//				std::ostringstream oss;
//				oss<<"Neuron "<<nrni++<<" (from SWC)";
//				np->name=oss.str();
//				trees.push_back(n);
//			}
//			if(degree[idx]==0) {
//				auto v1=VertexPriv::alloc();
//				auto vp1=VertexPriv::get(v1);
//				vp1->finished=true;
//				vertices.push_back(v1);
//				auto e=EdgePriv::alloc();
//				auto ep=EdgePriv::get(e);
//				ep->leftVertex=v;
//				vp->neighbors.push_back({e, false});
//				ep->rightVertex=v1;
//				vp1->neighbors.push_back({e, true});
//				ep->type=0;
//				auto t=std::get<1>(swc[idx]);
//				if(t>=0 && t<5) {
//					ep->type=t;
//					t=0;
//				} else if(t<0) {
//					t=0;
//				} else {
//					t=t-4;
//				}
//				ep->points.push_back(Point{std::get<2>(swc[idx]), std::get<3>(swc[idx]), std::get<4>(swc[idx]), std::get<5>(swc[idx]), t});
//				ep->points.push_back(Point{std::get<2>(swc[idx]), std::get<3>(swc[idx]), std::get<4>(swc[idx]), std::get<5>(swc[idx]), t});
//				edges.push_back(e);
//			}
//		}
//	}
//	for(size_t idx=0; idx<N; idx++) {
//		auto id=std::get<0>(swc[idx]);
//		auto par=std::get<6>(swc[idx]);
//		if(par!=-1 && degree[idx]!=2) {
//			auto e=EdgePriv::alloc();
//			auto ep=EdgePriv::get(e);
//			ep->leftVertex=id2vert[id];
//			auto vp=VertexPriv::get(ep->leftVertex);
//			vp->neighbors.push_back({e, false});
//			Vertex vert{};
//			auto i=idx;
//			ep->points.push_back(Point{std::get<2>(swc[i]), std::get<3>(swc[i]), std::get<4>(swc[i]), std::get<5>(swc[i]), std::get<1>(swc[i])});
//			do {
//				i=id2idx[par];
//				ep->points.push_back(Point{std::get<2>(swc[i]), std::get<3>(swc[i]), std::get<4>(swc[i]), std::get<5>(swc[i]), std::get<1>(swc[i])});
//				vert=id2vert[par];
//				par=std::get<6>(swc[i]);
//			} while(!vert);
//			ep->rightVertex=vert;
//			vp=VertexPriv::get(ep->rightVertex);
//			vp->neighbors.push_back({e, true});
//			std::array<int, 5> types;
//			for(auto& v: types) v=0;
//			for(auto& pt: ep->points) {
//				if(pt.m>=0 && pt.m<5) {
//					types[pt.m]++;
//					pt.m=0;
//				} else if(pt.m<0) {
//					pt.m=0;
//				} else {
//					pt.m=pt.m-4;
//				}
//			}
//			ep->type=0;
//			int max=types[0];
//			for(size_t i=2; i<types.size(); i++) {
//				if(types[i]>max) {
//					max=types[i];
//					ep->type=i;
//				}
//			}
//			edges.push_back(e);
//		}
//	}
//	updateModel();
//}
//
//void GraphPriv::fromSwc(const char * filename)
//{
//	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> swc{};
//
//	try {
//
//		std::ifstream ifs{ filename };
//		if (!ifs)
//			std::cout  << "File open failed : " << filename ;
//		ifs.precision(13);
//		ifs.setf(std::ios::scientific);
//
//		std::string line;
//		while (std::getline(ifs, line)) {
//			if (line.size() == 0) continue;
//			if (line[0] == '#') continue;
//			std::istringstream iss{ line };
//			iss.precision(13);
//			iss.setf(std::ios::scientific);
//
//			int64_t id, par;
//			int16_t type;
//			double x, y, z, r;
//			iss >> id >> type >> x >> y >> z >> r >> par;
//			if (!iss)
//				std::cout  << "Failed to parse line: " << line ;
//			swc.push_back(std::make_tuple(id, type, x, y, z, r, par));
//		}
//		if (!ifs.eof())
//			std::cout << "Failed to read file." ;
//	}
//	catch (std::exception& e) {
//		std::cout << e.what() ;
//
//	}
//
//	if (swc.size() == 0) {
//		std::cerr << "Empty SWC file.\n";
//		return ;
//	}
//	fromSwc(swc);
//}


