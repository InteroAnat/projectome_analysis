#ifndef _FNT_UTILS_H_
#define _FNT_UTILS_H_

#include <sstream>
#include <stdexcept>
#include <iostream>

#if defined(_WIN32) 
#include <windows.h>
#else
#ifdef BUILD_APPLE
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#endif

inline void printToStream(std::ostream& s) { }
template<typename Arg, typename... Args>
inline void printToStream(std::ostream& s, Arg&& a, Args&&... args) {
	return printToStream(s<<std::forward<Arg>(a), std::forward<Args>(args)...);
}

template<typename... Args>
inline void printMessage(Args&&... args) {
	return printToStream(std::cerr, std::forward<Args>(args)..., '\n');
}

inline void throwError() {
	throw std::runtime_error{"Unknown error"};
}
inline void throwError(const std::string& msg) {
	throw std::runtime_error{msg};
}
inline void throwError(const char* msg) {
	throw std::runtime_error{msg};
}
template<typename... Args>
inline void throwError(Args&&... args) {
	std::ostringstream oss;
	printToStream(oss, std::forward<Args>(args)...);
	throw std::runtime_error{oss.str()};
}

inline bool isLittleEndian() {
	int v=0x1234;
	auto p=reinterpret_cast<char*>(&v);
	return *p==0x34;
}

template <typename T, bool=std::is_floating_point<T>::value, bool=std::is_integral<T>::value, bool=std::is_signed<T>::value>
struct parseValue;
template <typename T>
struct parseValue<T, true, false, true> {
	inline T operator()(const char* str, char** endptr) { return strtod(str, endptr); }
};
template <typename T>
struct parseValue<T, false, true, true> {
	inline T operator()(const char* str, char** endptr) { return strtol(str, endptr, 10); }
};
template <typename T>
struct parseValue<T, false, true, false> {
	inline T operator()(const char* str, char** endptr) { return strtoul(str, endptr, 10); }
};

inline bool parseTuple(const char* str) {
	return str[0]=='\0';
}
template<typename T, typename... Args>
inline bool parseTuple(const char* str, T* vp, Args*... args) {
	char* endptr;
	T v=parseValue<T>{}(str, &endptr);
	if(endptr==str)
		return false;
	*vp=v;
	if(*endptr==':') {
		if(sizeof...(args)==0) {
			return false;
		} else {
			return parseTuple(endptr+1, args...);
		}
	} else if(*endptr=='\0') {
		if(sizeof...(args)==0) {
			return true;
		} else {
			return false;
		}
	} else {
		return false;
	}
}

inline int getNumberOfCpus() {
#ifdef WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
#else
#ifdef BUILD_APPLE
	int name[2];
	int numCPU;
	size_t len=sizeof(numCPU);
	name[0]=CTL_HW;
	name[1]=HW_NCPU;
	sysctl(name, 2, &numCPU, &len, nullptr, 0);
	return numCPU;
#else
	return sysconf(_SC_NPROCESSORS_ONLN);
#endif
#endif
}

inline std::string patternSubst(const std::string& pat, int32_t x, int32_t y, int32_t z) {
	std::ostringstream ss;
	for(size_t i=0; i<pat.size(); ) {
		auto cc=pat[i];
		if(cc!='<') {
			ss<<pat[i++];
			continue;
		}

		int state=0;
		char fill{' '};
		int width=0;
		int32_t val=0;

		size_t j;
		for(j=i+1; j<pat.size(); ) {
			auto c=pat[j];
			if(state==0) {
				fill=c;
				state=1;
				j++;
			} else if(state==1) {
				if(isdigit(c)) {
					width=width*10+(c-'0');
					j++;
					state=2;
				} else if(c=='>') {
					j++;
					state=6;
					break;
				} else {
					state=3;
				}
			} else if(state==2) {
				if(isdigit(c)) {
					width=width*10+(c-'0');
					j++;
				} else {
					state=3;
				}
			} else if(state==3) {
				if(c=='x' || c=='X') {
					val=x;
					j++;
					state=4;
				} else if(c=='y' || c=='Y') {
					val=y;
					j++;
					state=4;
				} else if(c=='z' || c=='Z') {
					val=z;
					j++;
					state=4;
				} else {
					break;
				}
			} else if(state==4) {
				if(c=='>') {
					state=5;
					j++;
					break;
				} else {
					break;
				}
			}
		}
		//
		if(state==5) {
			auto prev_f=ss.fill();
			auto prev_w=ss.width();
			ss.fill(fill);
			ss.width(width);
			ss<<val;
			ss.fill(prev_f);
			ss.width(prev_w);
			i=j;
		} else if(state==6) {
			ss<<fill;
			i=j;
		} else {
			ss<<cc;
			i++;
		}
	}
	return ss.str();
}


template<typename Stream, typename String>
Stream& fnt_getline(Stream& input, String& str) {
	auto& r=static_cast<Stream&>(std::getline(input, str));
	auto siz=str.size();
	if(siz>0 && str[siz-1]=='\r')
		str.resize(siz-1);
	return r;
}

#endif
