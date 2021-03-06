#include <iostream>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "stereoSGM.hpp"

using namespace std;
using namespace cv;

#define P1       5
#define P2       20
#define MAX_DISP 64

int main(int argc, char* argv[])
{
    String imgLeftPath;
    String imgRightPath;

    if(argc == 1)
    {
        imgLeftPath  = "../data/cones/im2.png";
        imgRightPath = "../data/cones/im6.png";
    }
    else if(argc == 3)
    {
        imgLeftPath  = argv[1];
        imgRightPath = argv[2];
    }
    else
    {
        cout << "Error: please input the left and right image!" << endl;
        exit(1);
    }
    cout << "start the stereoSGM test..." << endl;

    // ------------------------------------- //
    // get the left and right image          //
    // ------------------------------------- //
    Mat imgLeft  = imread(imgLeftPath,  CV_LOAD_IMAGE_GRAYSCALE);
    Mat imgRight = imread(imgRightPath, CV_LOAD_IMAGE_GRAYSCALE);

    // -------------------------------------- //
    // initial the stereo engine with image   //
    // geometry, max disparity and penalties  //
    // then start to compute the disparity    //
    // map by inputing the left and right bin //
    // -------------------------------------- //
    stereoSGM sgmEngine(imgLeft.cols, imgLeft.rows, MAX_DISP, P1, P2);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    sgmEngine.compute(imgLeft.data, imgRight.data);

    gettimeofday(&end, NULL);
    float elapsedTime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6f;
    cout << "compute elapsed time = " << to_string(elapsedTime) << "s" << endl;

    // -------------------------------------- //
    // display the result for visual check    //
    // 1. make the invalid as 0               //
    // 2. normalize the result to [0.0, 1.0)  //
    // -------------------------------------- //
    float   *dispFlt = new float[imgLeft.cols * imgLeft.rows];
    uint8_t *dispInt = new uint8_t[imgLeft.cols * imgLeft.rows];

    uint8_t max = 0;
    for(int i = 0; i < imgLeft.cols * imgLeft.rows; i++)
    {
        int8_t val = *(sgmEngine.m_dispMap + i);
        if(val == -1) val = 0;
        if(val > max) max = val;
        *(dispFlt + i) = static_cast<float>(val);
    }

    for(int i = 0; i < imgLeft.cols * imgLeft.rows; i++)
    {
        *(dispInt + i) = static_cast<uint8_t>(*(dispFlt + i) / static_cast<float>(max) * 255.0f);
    }

    Mat dispImage = Mat(imgLeft.rows, imgLeft.cols, CV_8U, dispInt);
    imwrite("./my_result.png", dispImage);
    imshow("disparity map", dispImage);
    waitKey(0);

    delete []dispFlt;
    delete []dispInt;
    return 0;
}
