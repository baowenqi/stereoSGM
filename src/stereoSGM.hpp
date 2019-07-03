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

    stereoSGMCostCube(int32_t i_cubeWidth, int32_t i_cubeHeight, int32_t i_cubeDisp, int32_t i_cubeBoundaryVal) :
    m_cubeWidth(i_cubeWidth), m_cubeHeight(i_cubeHeight), m_cubeDisp(i_cubeDisp), m_cubeBoundaryVal(i_cubeBoundaryVal)
    {
        m_cubeLP = m_cubeWidth * m_cubeDisp;
        m_cube = new T[m_cubeHeight * m_cubeLP];
        std::fill_n(m_cube, 0, m_cubeHeight * m_cubeLP * sizeof(T));
    }

   ~stereoSGMCostCube()
    { delete []m_cube; }

    // --------------------------------------------- //                                          
    // if coordinate(x, y, d) is out-of-boundary,    //
    // get method return a constant value predefined //
    // set method dummy the operation                //
    // --------------------------------------------- //
    T get(int32_t x, int32_t y, int32_t d)
    {
        if(x >= 0 && x < m_cubeWidth &&
           y >= 0 && y < m_cubeHeight &&
           d >= 0 && d < m_cubeDisp)
           return *(m_cube + y * m_cubeLP + x * m_cubeDisp + d);
        else
           return m_cubeBoundaryVal;
    }
    
    void set(int32_t x, int32_t y, int32_t d, T val)
    {
        if(x >= 0 && x < m_cubeWidth &&
           y >= 0 && y < m_cubeHeight &&
           d >= 0 && d < m_cubeDisp)
           *(m_cube + y * m_cubeLP + x * m_cubeDisp + d) = val;
    }

    // --------------------------------------------- //                                          
    // find the mininum cost in disp column(x, y)    //
    // --------------------------------------------- //                                          
    T getMin(int32_t x, int32_t y)
    {
        T min = m_cubeBoundaryVal;
        if(x >= 0 && x < m_cubeWidth && y >= 0 && y < m_cubeHeight)
        {
            int cubeOfst = y * m_cubeLP + x * m_cubeDisp;
            for(int d = 0; d < m_cubeDisp; d++)
            {
                T val = *(m_cube + cubeOfst + d);
                if(val < min) min = val;
            }
        }

        return min;
    }

    stereoSGMCostCube& operator+=(const stereoSGMCostCube<T>& rhs)
    {
        for(int i = 0; i < m_cubeHeight * m_cubeLP; i++)
        {
            *(this->m_cube + i) += static_cast<T>(*(rhs.m_cube + i));
        }
        return *this;
    }
};

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

