// neapi.cpp
//
// http://numenta.org/docs/nupic.core/

//
//#define NTA_PLATFORM_linux64
#define NTA_PLATFORM_darwin64

#include <iostream>

#include <nta/engine/Network.hpp>
#include <nta/engine/NuPIC.hpp>
#include <nta/engine/Region.hpp>
#include <nta/engine/YAMLUtils.hpp>
#include <nta/ntypes/Dimensions.hpp>


using namespace std;
using namespace nta;

Network make_network(){
    cout << "## make network" << endl;

    // input
    Dimensions d;
    d.push_back(4);
    d.push_back(4);
    cout << " input :  " << d[0]  << endl;
    cout << " input :  " << d[1]  << endl;
    
    Network net;

    // region 1
    Region *l1 = net.addRegion("level1", "TestNode", "");
    l1->setDimensions(d);

    std::set<UInt32> phases = net.getPhases("level1");

    // region 2
    net.addRegion("level2", "TestNode", "");
    phases = net.getPhases("level2");

    // link
    net.link("level1", "level2", "TestFanIn2", "");


    // initialize
    net.initialize();

    // execute
    net.run(1);

    // output
    Region* l2 = net.getRegions().getByName("level2");
    Dimensions d2 = l2->getDimensions();
    cout << " output :  " << d2[0]  << endl;
    cout << " output :  " << d2[1]  << endl;

    return net;
}

int main()
{
    // inheritance
    cout << "## inheritance" << endl;

    Network net = make_network();
}

