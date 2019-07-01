#include <iostream>
#include "stereoSGM.hpp"

using namespace std;

int main()
{
    cout << "start the stereoSGM test..." << endl;

    stereoSGM sgmEngine(640, 480, 64, 3, 20);
    stereoSGMCostCube<int8_t> matchCost(640, 480, 64, 64);

    return 0;
}
