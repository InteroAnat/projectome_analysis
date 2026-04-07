#ifndef _FNT_GRAPH_H_
#define _FNT_GRAPH_H_

#include <vector>
#include <utility>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <string>

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

#endif

