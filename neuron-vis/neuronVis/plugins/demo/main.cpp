#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <tuple>
#include "../SwcLoader/SwcLoader.h"
#include "../SwcLoader/graph.priv.h"
int main(){
    
    clock_t startime, endtime;
    startime=clock();//记录开始时间
    std::string testFileName="/Users/wxf/project/neuron-vis/resource/swc/202562/009.swc";
	neuronVis::SwcLoader loader;
    auto neuron=loader.ReadSWC(testFileName);
    std::cout<<neuron.edges.size()<<std::endl;
    endtime=clock();//记录结束时间
    double tot_time = (double)(endtime - startime);
    std::cout<<tot_time/1000<<std::endl;




	return 0;
}
