#pragma once
#include <stdint.h>
#include <cstdio>

// ------------------------------------------------- //
// a customized structure providing the abstraction  //
// to the cost cubes used by stereo engine           //
// ------------------------------------------------- //
template<typename T>
struct stereoSGMCostCube
{
    int32_t  m_cubeWidth;
    int32_t  m_cubeHeight;
    int32_t  m_cubeDisp;
    int32_t  m_cubeLP;
    T        m_cubeBoundaryVal;
    T       *m_cube;

    stereoSGMCostCube(int32_t i_cubeWidth, int32_t i_cubeHeight, int32_t i_cubeDisp, int32_t i_cubeBoundaryVal);
   ~stereoSGMCostCube();
    T get(int32_t x, int32_t y, int32_t d);
    void set(int32_t x, int32_t y, int32_t d, T val);
    T getMin(int32_t x, int32_t y);
    stereoSGMCostCube& operator+=(const stereoSGMCostCube<T>& rhs);
};

// ------------------------------------------------- //
// this is the stereoSGM engine class                //
// initial the engine by instance the class and call //
// compute to get the disparity map.                 //
// ------------------------------------------------- //
class stereoSGM
{
    public:
    int8_t  *m_dispMap;
    typedef enum e_status {SUCCESS, ERROR} status_t;

    // --------------------------------------------- //                                          
    // public method                                 //
    // after using constructor to config the engine, //
    // call compute the get the disparity map.       //
    // --------------------------------------------- //
             stereoSGM(int32_t i_imgWidth, int32_t i_imgHeight, int32_t i_imgDisp, int8_t i_P1, int8_t i_P2);
            ~stereoSGM();
    status_t compute(uint8_t *imgLeft, uint8_t *imgRight);

    private:
    // --------------------------------------------- //                                          
    // class constant variables:                     //
    // in this example, we fix the window size as 5  //                                          
    // so the useful output bits is 24               //                                          
    // --------------------------------------------- //
    const int32_t ctWinSize = 5;
    const int32_t ctWinRad = (ctWinSize - 1) / 2;
    const int32_t ctDataLen = ctWinSize * ctWinSize - 1;
    const int32_t ctDataMsb = ctDataLen - 1;
    const int8_t  lrcInvalid = -1;
    const int8_t  lrcThreshold = 1;

    // --------------------------------------------- //                                          
    // private variables                             //
    // --------------------------------------------- //                                          
    int32_t  m_imgWidth;
    int32_t  m_imgHeight;
    int32_t  m_imgDisp;
    int8_t   m_P1;
    int8_t   m_P2;
    typedef enum e_direction {L0, L1, L2, L3, L4, L5, L6, L7} direction_t;
    typedef enum e_dispmap {LEFT, RIGHT} dispmap_t;

    // --------------------------------------------- //                                          
    // private methods                               //
    // --------------------------------------------- //
    status_t f_censusTransform5x5(uint8_t *src, int32_t *dst);
    int32_t  f_getHammingDistance(int32_t src1, int32_t src2);
    status_t f_getMatchCost(int32_t *ctLeft, int32_t *ctRight, stereoSGMCostCube<int32_t> &matchCost);
    template<direction_t dir>
    status_t f_getPathCost(stereoSGMCostCube<int32_t> &matchCost, stereoSGMCostCube<int32_t> &pathCost);
    status_t f_aggregateCost(stereoSGMCostCube<int32_t> &matchCost, stereoSGMCostCube<int32_t> &sumCost);
    template<dispmap_t lr>
    status_t f_pickDisparity(stereoSGMCostCube<int32_t> &sumCost, int8_t *dispMap);
    status_t f_checkLeftRight(int8_t *dispLeft, int8_t *dispRight);
    status_t f_medianFilter3x3(int8_t *src, int8_t *dst);
};

