#include "graph.priv.h"
#include "utils.h"
#include "config-fnt.h"
//#include "stream.h"

#include <queue>
#include <tuple>
#include <unordered_map>
#include <fstream>
#include <array>

void GraphPriv::load(std::istream& fs) {
	size_t nv;
	fs>>nv;
	if(!fs.good())
		throwError("Failed to read size_t");
	vertices.resize(nv);
	for(size_t i=0; i<nv; i++) {
		vertices[i]=VertexPriv::alloc();
		auto vp=VertexPriv::get(vertices[i]);
		int f;
		fs>>f;
		vp->finished=f;
		if(!fs.good())
			throwError("Failed to read bool");
		vp->index=i;
	}

	size_t ne;
	fs>>ne;
	if(!fs.good())
		throwError("Failed to read size_t");
	edges.resize(ne);
	for(size_t i=0; i<ne; i++) {
		edges[i]=EdgePriv::alloc();
		auto ep=EdgePriv::get(edges[i]);
		size_t eli, eri, epl;
		fs>>ep->type;
		fs>>eli;
		fs>>eri;
		fs>>epl;
		if(!fs.good())
			throwError("Failed to read ints");
		ep->leftVertex=vertices[eli];
		VertexPriv::get(vertices[eli])->neighbors.push_back({edges[i], false});
		ep->rightVertex=vertices[eri];
		VertexPriv::get(vertices[eri])->neighbors.push_back({edges[i], true});
		ep->points.resize(epl);
		for(size_t j=0; j<epl; j++) {
			auto& p=ep->points[j];
			fs>>p.m>>p._x>>p._y>>p._z>>p._r;
			if(!fs.good())
				throwError("Failed to read point");
		}
		ep->index=i;
		ep->vaoi=-1;
	}

	size_t nn;
	fs>>nn;
	/**/
	if(!fs.good())
		throwError("Failed to read size_t");
	trees.resize(nn);
	for(size_t i=0; i<nn; i++) {
		trees[i]=TreePriv::alloc();
		auto np=TreePriv::get(trees[i]);
		size_t si;
		fs>>si;
		fnt_getline(std::ws(fs), np->name);
		if(!fs.good())
			throwError("Failed to read tree");
		np->root=vertices[si];
		np->index=i;
	}
}
void GraphPriv::save(std::ostream& fs) const {
	fs<<vertices.size()<<'\n';
	if(!fs.good())
		throwError("Failed to write header");
	for(size_t i=0; i<vertices.size(); i++) {
		auto v=vertices[i];
		fs<<v.finished()<<'\n';
		if(!fs.good())
			throwError("Failed to write header");
	}

	fs<<edges.size()<<'\n';
	if(!fs.good())
		throwError("Failed to write header");
	for(size_t i=0; i<edges.size(); i++) {
		auto e=edges[i];
		fs<<e.type()<<' ';
		fs<<e.leftVertex().index()<<' ';
		fs<<e.rightVertex().index()<<' ';
		fs<<e.points().size()<<'\n';
		if(!fs.good())
			throwError("Failed to write header");
		for(size_t j=0; j<e.points().size(); j++) {
			auto& p=e.points()[j];
			fs<<p.m<<' '<<p._x<<' '<<p._y<<' '<<p._z<<' '<<p._r<<'\n';
			if(!fs.good())
				throwError("Failed to write header");
		}
	}

	fs<<trees.size()<<'\n';
	if(!fs.good())
		throwError("Failed to write header");
	for(size_t i=0; i<trees.size(); i++) {
		auto n=trees[i];
		fs<<n.root().index()<<' '<<n.name()<<'\n';
		if(!fs.good())
			throwError("Failed to write header");
	}
}
void GraphPriv::updateModel() {
	for(size_t i=0; i<vertices.size(); i++) {
		auto vp=VertexPriv::get(vertices[i]);
		vp->index=i;
		vp->tree=Tree{};
		vp->inLoop=false;
		vp->parentVertex=Vertex{};
		vp->parentEdge=Edge{};
	}
	for(size_t i=0; i<edges.size(); i++) {
		auto ep=EdgePriv::get(edges[i]);
		ep->index=i;
		ep->tree=Tree{};
		ep->inLoop=false;
		ep->parentVertex=Vertex{};
		ep->parentEdge=Edge{};
	}
	std::queue<Vertex> que{};
	for(size_t i=0; i<trees.size(); i++) {
		auto np=TreePriv::get(trees[i]);
		np->index=i;
		np->completed_prev=np->completed;
		np->completed=np->root.finished();
		auto vp=VertexPriv::get(np->root);
		vp->tree=trees[i];
		que.push(np->root);
	}
	while(!que.empty()) {
		auto v=que.front();
		auto vp=VertexPriv::get(v);
		que.pop();
		auto vpnp=TreePriv::get(vp->tree);
		for(auto& p: vp->neighbors) {
			auto ep=EdgePriv::get(p.first);
			if(ep->tree)
				continue;
			/*
			if(vp->parentEdge)
				ep->type=vp->parentEdge.type();
				*/
			auto nv=ep->leftVertex==v ? ep->rightVertex : ep->leftVertex;
			auto nvp=VertexPriv::get(nv);
			ep->tree=vp->tree;
			ep->parentEdge=vp->parentEdge;
			ep->parentVertex=v;
			if(!nvp->finished)
				vpnp->completed=false;
			if(!nvp->tree) {
				nvp->tree=vp->tree;
				nvp->parentVertex=v;
				nvp->parentEdge=p.first;
				que.push(nv);
				continue;
			}

			// loop found
			auto nvnp=TreePriv::get(nvp->tree);
			vpnp->completed=false;
			nvnp->completed=false;
			nvp->inLoop=true;
			ep->inLoop=true;
			{
				auto nep=EdgePriv::get(nv.parentEdge());
				if(nep)
					nep->inLoop=true;
			}
			std::vector<Vertex> pars{};
			for(auto p=nvp->parentVertex; p; p=p.parentVertex()) {
				pars.emplace_back(p);
			}
			Vertex commp{};
			for(auto p=v; p; p=p.parentVertex()) {
				for(auto pp: pars) {
					if(pp==p) {
						commp=p;
						break;
					}
				}
				if(commp)
					break;
			}
			for(auto pp: pars) {
				auto ppvp=VertexPriv::get(pp);
				ppvp->inLoop=true;
				if(pp==commp)
					break;
				{
					auto nep=EdgePriv::get(pp.parentEdge());
					if(nep)
						nep->inLoop=true;
				}
			}
			for(auto p=v; p; p=p.parentVertex()) {
				auto pvp=VertexPriv::get(p);
				pvp->inLoop=true;
				if(p==commp)
					break;
				{
					auto nep=EdgePriv::get(p.parentEdge());
					if(nep)
						nep->inLoop=true;
				}
			}
		}
	}
}
//
//std::vector<std::string> GraphPriv::loadFnt(const char* filename) {
//	std::vector<std::string> header{};
//	InputStream fs;
//	if(!fs.open(filename))
//		throwError("Failed to open file: ", filename);
//
//	fs.precision(13);
//	fs.setf(std::ios::scientific);
//
//	std::string line;
//	fnt_getline(fs, line);
//	if(!fs)
//		throwError("Failed to read magic header");
//
//	bool gzipped=false;
//	if(line==FNTZ_MAGIC) {
//		gzipped=true;
//	} else if(line!=FNT_MAGIC) {
//		throwError("Wrong magic: ", line);
//	}
//	header.push_back(line);
//
//	fnt_getline(fs, line);
//	if(!fs)
//		throwError("Failed to read URL line");
//	while(line!="BEGIN_TRACING_DATA") {
//		header.push_back(line);
//		fnt_getline(fs, line);
//		if(!fs)
//			throwError("Failed to read URL line");
//	}
//
//	if(gzipped) {
//		if(!fs.pushGzip())
//			throwError("Failed to start decompressing");
//	}
//	load(fs);
//	if(gzipped) {
//		if(!fs.pop())
//			throwError("Failed to finish decompressing");
//	}
//
//	if(!fs.close())
//		throwError("Failed to close file.");
//	updateModel();
//
//	return header;
//}
//void GraphPriv::saveFnt(const char* filename, const std::vector<std::string>& header) const {
//	OutputStream fs;
//	if(!fs.open(filename))
//		throwError("Failed to open file: ", filename);
//	fs.precision(13);
//	fs.setf(std::ios::scientific);
//
//	bool gzipped=false;
//	if(header[0]==FNTZ_MAGIC)
//		gzipped=true;
//	for(auto& l: header) {
//		fs<<l<<'\n';
//		if(!fs)
//			throwError("Failed to write line");
//	}
//	fs<<"BEGIN_TRACING_DATA"<<'\n';
//	if(!fs)
//		throwError("Failed to write data start mark");
//
//	if(gzipped) {
//		if(!fs.pushGzip())
//			throwError("Failed to start compressing");
//	}
//	save(fs);
//	if(gzipped) {
//		if(!fs.pop())
//			throwError("Failed to finish compressing");
//	}
//
//	if(!fs.close())
//		throwError("Failed to close file");
//}


std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> GraphPriv::toSwc() const {
	std::vector<int64_t> v2nid(vertices.size(), -1);
	std::vector<bool> edgemask(edges.size(), false);
	std::queue<Vertex> que{};
	int64_t nid=1;
	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> swc;
	while(true) {
		if(que.empty()) {
			for(auto n: trees) {
				auto s=n.root();
				if(v2nid[s.index()]==-1) {
					auto& pos=s.point();
					swc.push_back(std::make_tuple(nid, 1, pos.x(), pos.y(), pos.z(), pos.r(), -1));
					v2nid[s.index()]=nid++;
					que.push(s);
				}
			}
		}
		if(que.empty()) {
			for(auto v: vertices) {
				if(v2nid[v.index()]==-1) {
					auto& pos=v.point();
					swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:v.neighbors()[0].first.type(), pos.x(), pos.y(), pos.z(), pos.r(), -1));
					v2nid[v.index()]=nid++;
					que.push(v);
					break;
				}
			}
		}
		if(que.empty())
			break;

		while(!que.empty()) {
			auto v=que.front();
			que.pop();
			auto nid_par=v2nid[v.index()];
			for(auto& ep: v.neighbors()) {
				auto e=ep.first;
				if(edgemask[e.index()])
					continue;
				if(e.length()==2 && e.points()[0]==e.points()[1]) {
					Vertex v1;
					if(ep.second) {
						v1=e.leftVertex();
					} else {
						v1=e.rightVertex();
					}
					if(v1.neighbors().size()==1) {
						edgemask[e.index()]=true;
						if(v2nid[v1.index()]==-1) {
							v2nid[v1.index()]=nid++;
						}
						continue;
					}
				}
				Vertex v1;
				auto pnid=nid_par;
				if(ep.second) {
					for(size_t j=e.length()-1; j-->0;) {
						auto& pos=e.points()[j];
						swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:e.type(), pos.x(), pos.y(), pos.z(), pos.r(), pnid));
						pnid=nid++;
					}
					v1=e.leftVertex();
				} else {
					for(size_t j=1; j<e.length(); j++) {
						auto& pos=e.points()[j];
						swc.push_back(std::make_tuple(nid, pos.m?pos.m+9:e.type(), pos.x(), pos.y(), pos.z(), pos.r(), pnid));
						pnid=nid++;
					}
					v1=e.rightVertex();
				}
				edgemask[e.index()]=true;
				if(v2nid[v1.index()]==-1) {
					v2nid[v1.index()]=pnid;
					que.push(v1);
				}
			}
		}
	}
	return swc;
}

void GraphPriv::fromSwc(const std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>>& swc) {
	auto N=swc.size();
	std::vector<int> degree(N, 0);
	std::unordered_map<int64_t, size_t> id2idx;

	for(size_t idx=0; idx<N; idx++) {
		auto id=std::get<0>(swc[idx]);
		auto par=std::get<6>(swc[idx]);
		auto i=id2idx.find(id);
		if(i!=id2idx.end())
			throwError("Duplicated nodes.");
		id2idx[id]=idx;
		if(par!=-1) {
			auto j=id2idx.find(par);
			if(j==id2idx.end())
				throwError("Node refered but not defined yet.");
			degree[idx]++;
			degree[j->second]++;
		}
	}

	int nrni=1;
	std::unordered_map<int64_t, Vertex> id2vert;
	for(size_t idx=0; idx<N; idx++) {
		auto id=std::get<0>(swc[idx]);
		auto par=std::get<6>(swc[idx]);
		bool node=false;
		bool soma=false;
		if(par==-1) {
			node=true;
			if(std::get<1>(swc[idx])==1) {
				soma=true;
			}
		} else {
			switch(degree[idx]) {
				case 2:
					break;
				default:
					node=true;
					break;
			}
		}
		if(node) {
			auto v=VertexPriv::alloc();
			auto vp=VertexPriv::get(v);
			vp->finished=true;
			id2vert[id]=v;
			vertices.push_back(v);
			if(soma) {
				auto n=TreePriv::alloc();
				auto np=TreePriv::get(n);
				np->root=v;
				std::ostringstream oss;
				oss<<"Neuron "<<nrni++<<" (from SWC)";
				np->name=oss.str();
				trees.push_back(n);
			}
			if(degree[idx]==0) {
				auto v1=VertexPriv::alloc();
				auto vp1=VertexPriv::get(v1);
				vp1->finished=true;
				vertices.push_back(v1);
				auto e=EdgePriv::alloc();
				auto ep=EdgePriv::get(e);
				ep->leftVertex=v;
				vp->neighbors.push_back({e, false});
				ep->rightVertex=v1;
				vp1->neighbors.push_back({e, true});
				ep->type=0;
				auto t=std::get<1>(swc[idx]);
				if(t>=0 && t<5) {
					ep->type=t;
					t=0;
				} else if(t<0) {
					t=0;
				} else {
					t=t-4;
				}
				ep->points.push_back(Point{std::get<2>(swc[idx]), std::get<3>(swc[idx]), std::get<4>(swc[idx]), std::get<5>(swc[idx]), t});
				ep->points.push_back(Point{std::get<2>(swc[idx]), std::get<3>(swc[idx]), std::get<4>(swc[idx]), std::get<5>(swc[idx]), t});
				edges.push_back(e);
			}
		}
	}
	for(size_t idx=0; idx<N; idx++) {
		auto id=std::get<0>(swc[idx]);
		auto par=std::get<6>(swc[idx]);
		if(par!=-1 && degree[idx]!=2) {
			auto e=EdgePriv::alloc();
			auto ep=EdgePriv::get(e);
			ep->leftVertex=id2vert[id];
			auto vp=VertexPriv::get(ep->leftVertex);
			vp->neighbors.push_back({e, false});
			Vertex vert{};
			auto i=idx;
			ep->points.push_back(Point{std::get<2>(swc[i]), std::get<3>(swc[i]), std::get<4>(swc[i]), std::get<5>(swc[i]), std::get<1>(swc[i])});
			do {
				i=id2idx[par];
				ep->points.push_back(Point{std::get<2>(swc[i]), std::get<3>(swc[i]), std::get<4>(swc[i]), std::get<5>(swc[i]), std::get<1>(swc[i])});
				vert=id2vert[par];
				par=std::get<6>(swc[i]);
			} while(!vert);
			ep->rightVertex=vert;
			vp=VertexPriv::get(ep->rightVertex);
			vp->neighbors.push_back({e, true});
			std::array<int, 5> types;
			for(auto& v: types) v=0;
			for(auto& pt: ep->points) {
				if(pt.m>=0 && pt.m<5) {
					types[pt.m]++;
					pt.m=0;
				} else if(pt.m<0) {
					pt.m=0;
				} else {
					pt.m=pt.m-4;
				}
			}
			ep->type=0;
			int max=types[0];
			for(size_t i=2; i<types.size(); i++) {
				if(types[i]>max) {
					max=types[i];
					ep->type=i;
				}
			}
			edges.push_back(e);
		}
	}
	updateModel();
}

void GraphPriv::fromSwc(const char * filename)
{
	std::vector<std::tuple<int64_t, int16_t, double, double, double, double, int64_t>> swc{};

	try {

		std::ifstream ifs{ filename };
		if (!ifs)
			std::cout << "File open failed : " << filename<< std::endl;
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
				std::cout << "Failed to parse line: " << line <<std::endl;
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
		return ;
	}
	fromSwc(swc);
}

const char* FNT_MAGIC=PACKAGE_NAME_LONG " Session File 1.0";
const char* FNTZ_MAGIC=PACKAGE_NAME_LONG " Session File 1.0 (gzipped)";
