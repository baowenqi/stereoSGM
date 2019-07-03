#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "stereoSGM.hpp"

using namespace std;
using namespace cv;

#define P1       5
#define P2       20
#define MAX_DISP 64

int main()
{
    cout << "start the stereoSGM test..." << endl;

    Mat imgLeft  = imread("../data/cones/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat imgRight = imread("../data/cones/im6.png", CV_LOAD_IMAGE_GRAYSCALE);

    Mat padLeft, padRight;
    copyMakeBorder(imgLeft, padLeft, 2, 2, 2, 2, BORDER_CONSTANT, 0);
    copyMakeBorder(imgRight, padRight, 2, 2, 2, 2, BORDER_CONSTANT, 0);

    imshow("test", padLeft);
    waitKey(0);
    imshow("test", padRight);
    waitKey(0);

    stereoSGM sgmEngine(imgLeft.cols, imgLeft.rows, MAX_DISP, P1, P2);
    sgmEngine.compute(padLeft.data, padRight.data);

    // -------------------------------------- //
    // display the result for visual check    //
    // make the invalid as 0                  //
    // normalize the result                   //
    // -------------------------------------- //
    float *display = new float[imgLeft.cols * imgLeft.rows];

    uint8_t max = 0;
    for(int i = 0; i < imgLeft.cols * imgLeft.rows; i++)
    {
        int8_t val = *(sgmEngine.m_dispMap + i);
        if(val == -1) val = 0;
        if(val > max) max = val;
        *(display + i) = static_cast<float>(val);
    }

    for(int i = 0; i < imgLeft.cols * imgLeft.rows; i++)
    {
        *(display + i) /= static_cast<float>(max);
    }

    Mat dispImage = Mat(imgLeft.rows, imgLeft.cols, CV_32F, display);
    imshow("test", dispImage);
    waitKey(0);

    return 0;
}
