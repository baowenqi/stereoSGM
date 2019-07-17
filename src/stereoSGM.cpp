#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <sys/time.h>
#include "stereoSGM.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

// --------------------------------------------- //                                          
// Methods of stereoSGMCostCube structure        //
// --------------------------------------------- //                                          
template<typename T>
stereoSGMCostCube<T>::stereoSGMCostCube
(
    int32_t i_cubeWidth,
    int32_t i_cubeHeight,
    int32_t i_cubeDisp,
    int32_t i_cubeBoundaryVal
) :
m_cubeWidth(i_cubeWidth),
m_cubeHeight(i_cubeHeight),
m_cubeDisp(i_cubeDisp),
m_cubeBoundaryVal(i_cubeBoundaryVal)
{
    m_cubeLP = m_cubeWidth * m_cubeDisp;
    m_cube = new T[m_cubeHeight * m_cubeLP];
    std::fill_n(m_cube, 0, m_cubeHeight * m_cubeLP * sizeof(T));
}

template<typename T>
stereoSGMCostCube<T>::~stereoSGMCostCube(){ delete []m_cube; }

// --------------------------------------------- //                                          
// if coordinate(x, y, d) is out-of-boundary,    //
// get method return a constant value predefined //
// set method dummy the operation                //
// --------------------------------------------- //
template<typename T>
T stereoSGMCostCube<T>::get(int32_t x, int32_t y, int32_t d)
{
    if(x >= 0 && x < m_cubeWidth &&
       y >= 0 && y < m_cubeHeight &&
       d >= 0 && d < m_cubeDisp)
       return *(m_cube + y * m_cubeLP + x * m_cubeDisp + d);
    else
       return m_cubeBoundaryVal;
}

template<typename T>
void stereoSGMCostCube<T>::set(int32_t x, int32_t y, int32_t d, T val)
{
    if(x >= 0 && x < m_cubeWidth &&
       y >= 0 && y < m_cubeHeight &&
       d >= 0 && d < m_cubeDisp)
       *(m_cube + y * m_cubeLP + x * m_cubeDisp + d) = val;
}

// --------------------------------------------- //                                          
// find the mininum cost in disp column(x, y)    //
// --------------------------------------------- //                                          
template<typename T>
T stereoSGMCostCube<T>::getMin(int32_t x, int32_t y)
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

// --------------------------------------------- //                                          
// overload this operator to accumulate path     //
// costs to sum cost                             //
// --------------------------------------------- //
template<typename T>
stereoSGMCostCube<T>& stereoSGMCostCube<T>::operator+=(const stereoSGMCostCube<T>& rhs)
{
    for(int i = 0; i < m_cubeHeight * m_cubeLP; i++)
    {
        *(this->m_cube + i) += static_cast<T>(*(rhs.m_cube + i));
    }
    return *this;
}

// --------------------------------------------- //                                          
// Methods of stereoSGM class                    //
// --------------------------------------------- //                                          
stereoSGM::stereoSGM
(
    int32_t i_imgWidth,
    int32_t i_imgHeight,
    int32_t i_imgDisp,
    int8_t  i_P1,
    int8_t  i_P2
) :
m_imgWidth(i_imgWidth),
m_imgHeight(i_imgHeight),
m_imgDisp(i_imgDisp),
m_P1(i_P1),
m_P2(i_P2)
{
    // --------------------------------------------- //
    // the disparity range should be multiple of 16  //
    // in this example, disparity is 64              //
    // make sure P1 is smaller than P2 (constant)    //
    // --------------------------------------------- //
    assert(m_imgDisp % 16 == 0);
    assert(m_P1 < m_P2);

    m_dispMap = new int8_t[m_imgWidth * m_imgHeight];
}

stereoSGM::~stereoSGM()
{
    if(m_dispMap != nullptr) delete []m_dispMap;
}

// ------------------------------------------------------------------------------------------------------------ //
// this is kernel of the stereo engine, the SGM flow is like:                                                   //
// +-------+     +---------------+     +--------------+     +-----------------+                                 //
// | CT5x5 | ==> | compute       |     | compute      | ==> | pick left disp  | ==> +-------+                   //
// +-------+     | match cost    |     | 8 path costs |     +-----------------+     | left  |     +-----------+ //
//               | get cost cube | ==> | aggregate to |                             | right | ==> | median3x3 | //
// +-------+     | [H * W * D]   |     | sum cost     |     +-----------------+     | check |     +-----------+ //
// | CT5x5 | ==> |               |     | [H * W * D]  | ==> | pick right disp | ==> +-------+                   //
// +-------+     +---------------+     +--------------+     +-----------------+                                 //
//                                                                                                              //
// the function inputs are left and right buffers and output is in m_dispMap buffer.                            //
// ------------------------------------------------------------------------------------------------------------ //
stereoSGM::status_t stereoSGM::compute
(
    uint8_t *imgLeft,
    uint8_t *imgRight
)
{
    uint8_t *imgPad    = new uint8_t[(m_imgWidth + 4) * (m_imgHeight + 4)];
    int32_t *ctLeft    = new int32_t[m_imgWidth * m_imgHeight];
    int32_t *ctRight   = new int32_t[m_imgWidth * m_imgHeight];
    int8_t  *dispLeft  = new int8_t [m_imgWidth * m_imgHeight];
    int8_t  *dispRight = new int8_t [m_imgWidth * m_imgHeight];
    stereoSGMCostCube<int32_t> matchCost(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);
    stereoSGMCostCube<int32_t> sumCost(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);
    int8_t *dispLeftPad = new int8_t [(m_imgWidth + 2) * (m_imgHeight + 2)];

    f_padImage<uint8_t, CONSTANT>(imgLeft, imgPad, 2);
    f_censusTransform5x5(imgPad,  ctLeft);
    f_padImage<uint8_t, CONSTANT>(imgRight, imgPad, 2);
    f_censusTransform5x5(imgPad, ctRight);
    f_getMatchCost(ctLeft, ctRight, matchCost);
    f_aggregateCost(matchCost, sumCost);
    f_pickDisparity<LEFT>(sumCost, dispLeft);
    f_pickDisparity<RIGHT>(sumCost, dispRight);
    f_checkLeftRight(dispLeft, dispRight);
    f_padImage<int8_t, EXTEND>(dispLeft, dispLeftPad, 1);
    f_medianFilter3x3(dispLeft, m_dispMap);

    delete []imgPad;
    delete []ctLeft;
    delete []ctRight;
    delete []dispLeft;
    delete []dispRight;
    delete []dispLeftPad;

    return SUCCESS;
}

// ------------------------------------------------- //
// padding the src image with a certain bord pattern //
// currently support:                                //
// 1. CONSTANT                                       //
// 2. EXTEND                                         //
// ------------------------------------------------- //
template<typename T, stereoSGM::pad_t P>
stereoSGM::status_t stereoSGM::f_padImage
(
    T*  src,
    T*  dst,
    int padNum,
    T   constVal
)
{
    int dstWidth = m_imgWidth + 2 * padNum;
    int dstHeight = m_imgHeight + 2 * padNum;
    int dstOfst = dstWidth * padNum + padNum;

    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
	{
	    T srcData = *(src + y * m_imgWidth + x);
	    *(dst + dstOfst + y * dstWidth + x) = srcData;
	}
    }

    T padVal;
    if(P == CONSTANT) padVal = constVal;

    for(int y = 0; y < padNum; y++)
    {
        for(int x = padNum; x < dstWidth - padNum; x++)
        {
    	    if(P == EXTEND) padVal = *(dst + padNum * dstWidth + x);
            *(dst + y * dstWidth + x) = padVal;

            if(P == EXTEND) padVal = *(dst + (dstHeight - 1 - padNum) * dstWidth + x);
            *(dst + (dstHeight - 1 - y) * dstWidth + x) = padVal;
        }
    }
    
    for(int x = 0; x < padNum; x++)
    {
        for(int y = 0; y < dstHeight; y++)
        {
	    if(P == EXTEND) padVal = *(dst + y * dstWidth + padNum);
            *(dst + y * dstWidth + x) = padVal;

	    if(P == EXTEND) padVal = *(dst + y * dstWidth + (dstWidth - 1 - padNum));
            *(dst + y * dstWidth + (dstWidth - 1 - x)) = padVal;
        }
    }

    return SUCCESS;
}

// ------------------------------------------------- //
// census transform over a 5x5 window.               //
// it's caller's job to pad the border by 2 elements //
// a 5x5 window has 24 elements comparing with the   //
// center element so the output can be encoded in    //
// int32_t (lower 24 bits are useful).               //
// the mapping of the window elements position and   //
// output bit position (center-skipped) is like:     //
//                0    1    2    3    4              //
//             |----+----+----+----+----|            //
// bitPos1-> 0 | 23 | 22 | 21 | 20 | 19 |            // 
//             |----+----+----+----+----|            //
//           1 | 18 | 17 | 16 | 15 | 14 |            // 
//             |----+----+----+----+----|            //
//           2 | 13 | 12 |  * | 11 | 10 |            // 
//             |----+----+----+----+----|            //
//           3 |  9 |  8 |  7 |  6 |  5 |            // 
//             |----+----+----+----+----|            //
//           4 |  4 |  3 |  2 |  1 |  0 | <-bitPos0  // 
//             |----+----+----+----+----|            //
// if window(i, j) > window(2, 2), then its corres-  //
// ponding output bit is 1, otherwise it's 0.        //
// ------------------------------------------------- //
stereoSGM::status_t stereoSGM::f_censusTransform5x5
(
    uint8_t *src,
    int32_t *dst
)
{
    // --------------------------------------------- //
    // as the census transform window is 5x5, need   //
    // to pad 2 to the border so the src line pitch  //
    // is 4-element larger then dst                  // 
    // --------------------------------------------- //
    const int32_t dstLinePitch = m_imgWidth;
    const int32_t srcLinePitch = dstLinePitch + ctWinRad * 2;

    for(int dy = 0; dy < m_imgHeight; dy++)
    {
        for(int dx = 0; dx < m_imgWidth; dx++)
        {
            int32_t dstData = 0;
            
            // ------------------------------------- //
            // dx and dy map to top-left corner of   //
            // the window so that + winRad to get    //
            // center                                //
            // ------------------------------------- //
            int32_t srcCenterOfst = (dy + ctWinRad) * srcLinePitch + (dx + ctWinRad);
            uint8_t srcCenter = *(src + srcCenterOfst);

            // ------------------------------------- //
            // compute row0, row1 and row3, row4     //
            // bitPos0 counts from right-bottom      //
            // bitPos1 counts from left-top          //
            // ------------------------------------- //
            int32_t bitPos0 = 0, bitPos1 = ctDataMsb;
            for(int wy = 0; wy < ctWinRad; wy++)
            {
                for(int wx = 0; wx < ctWinSize; wx++, bitPos0++, bitPos1--)
                {
                    int32_t srcDataOfst0 = (dy + ctWinSize - 1 - wy) * srcLinePitch + (dx + ctWinSize - 1 -wx);
                    int32_t srcDataOfst1 = (dy + wy) * srcLinePitch + (dx + wx);

                    uint8_t srcData0 = *(src + srcDataOfst0);
                    uint8_t srcData1 = *(src + srcDataOfst1);

                    dstData |= (srcData0 > srcCenter) << bitPos0;
                    dstData |= (srcData1 > srcCenter) << bitPos1;
                }
            }
            // ------------------------------------- //
            // compute row2, col0, col1, col3, col4  //
            // so that we can skip center pixel      //
            // ------------------------------------- //
            for(int wx = 0; wx < ctWinRad; wx++, bitPos0++, bitPos1--)
            {
                int32_t srcDataOfst0 = (dy + ctWinRad) * srcLinePitch + (dx + ctWinSize - 1 -wx);
                int32_t srcDataOfst1 = (dy + ctWinRad) * srcLinePitch + (dx + wx);

                uint8_t srcData0 = *(src + srcDataOfst0);
                uint8_t srcData1 = *(src + srcDataOfst1);

                dstData |= (srcData0 > srcCenter) << bitPos0;
                dstData |= (srcData1 > srcCenter) << bitPos1;
            }
            *(dst + dy * m_imgWidth + dx) = dstData;
        }
    }

    return SUCCESS;
}

// ------------------------------------------------- //
// hamming distance is the number of different bits  //
// between src1 and src2. so we xor src1 and src2    //
// first and count the number of 1s. As the prestage //
// is 5x5 census transform, only 24 useful bits, the //
// max distance is 24 which can be store in 1 byte.  //
// but for the futher extension, like increase the   //
// ct window size or increase the disparity range,   //
// we store in 4 bytes (int32_t).                    //
// ------------------------------------------------- //
int32_t stereoSGM::f_getHammingDistance
(
    int32_t src1,
    int32_t src2
)
{
    int32_t diff = src1 ^ src2;

    int32_t distance = 0;
    for(int i = 0; i < ctDataLen; i++) distance += (diff >> i) & 1;

    return distance;
}

// ------------------------------------------------- //
// the pixel-wise matching cost is the hamming dist- //
// ance between L(x) and R(x+d), here d ranges from  //
// 0 to max disparity and forms a matching cost cube //
// matchCost(h * w * d), d is the inner-most dim.    //
// ------------------------------------------------- //
stereoSGM::status_t stereoSGM::f_getMatchCost
(
    int32_t *ctLeft,
    int32_t *ctRight,
    stereoSGMCostCube<int32_t>& matchCost
)
{
    const int32_t matchCostLP = m_imgWidth * m_imgDisp;

    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
        {
            int32_t leftData = *(ctLeft + y * m_imgWidth + x);
            for(int d = 0; d < m_imgDisp; d++)
            {
                int32_t rightData = 0;
                // ---------------------------------------- //
                // we need to make sure the right pixel is  //
                // not out-of-boundary, otherwise use zero  //
                // ---------------------------------------- //
                if(x - d >= 0) rightData = *(ctRight + y * m_imgWidth + x - d);
                matchCost.set(x, y, d, f_getHammingDistance(leftData, rightData));
            }
        }
    }
    return SUCCESS;
}

template<stereoSGM::direction_t dir>
stereoSGM::status_t stereoSGM::f_getPathCost
(
    stereoSGMCostCube<int32_t> &matchCost,
    stereoSGMCostCube<int32_t> &pathCost
)
{
    // ----------------------------------- //
    // initial the searching variables     //
    // according to different directions   //
    // ----------------------------------- //
    int dx, dy;
    switch(dir)
    {
        case(L0):
            dx = -1;
            dy =  0;
            break;
        case(L1):
            dx = -1;
            dy = -1;
            break;
        case(L2):
            dx =  0;
            dy = -1;
            break;
        case(L3):
            dx =  1;
            dy = -1;
            break;
        case(L4):
            dx =  1;
            dy =  0;
            break;
        case(L5):
            dx =  1;
            dy =  1;
            break;
        case(L6):
            dx =  0;
            dy =  1;
            break;
        case(L7):
            dx = -1;
            dy =  1;
            break;
        default:
            // fatal error
            return ERROR;
    }

    switch(dir)
    {
        // ----------------------------------- //
        // traverse from top-left to bot-right //
        // ----------------------------------- //
        case(L0): case(L1): case(L2): case(L3):
            for(int y = 0; y < m_imgHeight; y++)
            {
                for(int x = 0; x < m_imgWidth; x++)
                {
                    for(int d = 0; d < m_imgDisp; d++)
                    {
                        int32_t c    = matchCost.get(x,      y,      d);
                        int32_t la   = pathCost.get (x + dx, y + dy, d);
                        int32_t lb   = pathCost.get (x + dx, y + dy, d - 1) + m_P1;
                        int32_t lc   = pathCost.get (x + dx, y + dy, d + 1) + m_P1;
                        int32_t lmin = pathCost.getMin(x + dx, y + dy);
                        int32_t ld   = lmin + m_P2;
                        int32_t min4 = min(min(min(la, lb), lc), ld);

                        // ------------------------------------ //
                        // perform the key equation in SGM:     //
                        // Lr = C + min(Lr-1(d),                //
                        //              Lr-1(d-1)+P1,           // 
                        //              Lr-1(d+1)+P1,           //
                        //              min(Lr)+P2) - min(Lr)   //
                        // ------------------------------------ //
                        pathCost.set(x, y, d, c + min4 - lmin);
                    }
                }
            }
            break;
        // ----------------------------------- //
        // traverse from bot-right to top-left //
        // ----------------------------------- //
        case(L4): case(L5): case(L6): case(L7):
            for(int y = m_imgHeight - 1; y >= 0; y--)
            {
                for(int x = m_imgWidth - 1; x >= 0; x--)
                {
                    for(int d = 0; d < m_imgDisp; d++)
                    {
                        int32_t c    = matchCost.get(x,      y,      d);
                        int32_t la   = pathCost.get (x + dx, y + dy, d);
                        int32_t lb   = pathCost.get (x + dx, y + dy, d - 1) + m_P1;
                        int32_t lc   = pathCost.get (x + dx, y + dy, d + 1) + m_P1;
                        int32_t lmin = pathCost.getMin(x + dx, y + dy);
                        int32_t ld   = lmin + m_P2;
                        int32_t min4 = min(min(min(la, lb), lc), ld);

                        // ------------------------------------ //
                        // perform the key equation in SGM:     //
                        // Lr = C + min(Lr-1(d),                //
                        //              Lr-1(d-1)+P1,           // 
                        //              Lr-1(d+1)+P1,           //
                        //              min(Lr)+P2) - min(Lr)   //
                        // ------------------------------------ //
                        pathCost.set(x, y, d, c + min4 - lmin);
                    }
                }
            }
            break;
        default:
            // fatal error
            return ERROR;
    }

    return SUCCESS;
}

// ------------------------------------------------- //
// the aggregate function calculates and sums eight  //
// path cost, which defines as below:                //
// +------------------------------------------+      //
// |      L1\   |L2 /L3                       |      //
// |         \  |  /                          |      //
// |      L0  \ | /  L4                       |      //
// |     ------ * -------                     |      //
// |          / | \                           |      //
// |         /  |  \                          |      //
// |      L7/ L6|   \L5                       |      //
// +------------------------------------------+      //
// * is arbitrary pixel in the sum cost cube         //
// ------------------------------------------------- //
stereoSGM::status_t stereoSGM::f_aggregateCost
(
    stereoSGMCostCube<int32_t>  &matchCost,
    stereoSGMCostCube<int32_t> &sumCost
)
{
    stereoSGMCostCube<int32_t> *pathCostPtr[8];

    for(int i = 0; i < 8; i++)
        pathCostPtr[i] = new stereoSGMCostCube<int32_t>(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    f_getPathCost<L0>(matchCost, *(pathCostPtr[0]));
    gettimeofday(&end, NULL);
    float elapsedTime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6f;
    cout << "path cost elapsed time = " << to_string(elapsedTime) << "s" << endl;

    f_getPathCost<L1>(matchCost, *(pathCostPtr[1]));
    f_getPathCost<L2>(matchCost, *(pathCostPtr[2]));
    f_getPathCost<L3>(matchCost, *(pathCostPtr[3]));
    f_getPathCost<L4>(matchCost, *(pathCostPtr[4]));
    f_getPathCost<L5>(matchCost, *(pathCostPtr[5]));
    f_getPathCost<L6>(matchCost, *(pathCostPtr[6]));
    f_getPathCost<L7>(matchCost, *(pathCostPtr[7]));

    for(int i = 0; i < 8; i++)
    {
        sumCost += *(pathCostPtr[i]);
        delete pathCostPtr[i];
    }

    return SUCCESS;
}

template<stereoSGM::dispmap_t lr>
stereoSGM::status_t stereoSGM::f_pickDisparity
(
    stereoSGMCostCube<int32_t>& sumCost,
    int8_t *dispMap
)
{
    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
        {
            int32_t dispVal = m_imgDisp * 8;
            int8_t  dispId = 0;
            for(int d = 0; d < m_imgDisp; d++)
            {
                // ------------------------------------------ //
                // as sumCost aggregates 8 paths, the max val //
                // is max disparity * 8                       //
                // ------------------------------------------ //
                int32_t costVal = m_imgDisp * 8;

                if(lr == LEFT)  costVal = sumCost.get(x,     y, d);
                if(lr == RIGHT) costVal = sumCost.get(x + d, y, d);

                if(costVal < dispVal)
                {
                    dispVal = costVal;
                    dispId = d;
                }
            }
            *(dispMap + y * m_imgWidth + x) = dispId;
        }
    }

    return SUCCESS;
}

stereoSGM::status_t stereoSGM::f_checkLeftRight
(
    int8_t *dispLeft,
    int8_t *dispRight
)
{
    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
        {
            int8_t dl = *(dispLeft + y * m_imgWidth + x);

            int xr = x - static_cast<int>(dl);

            int8_t dr = m_imgDisp;
            if(xr >= 0)
                dr = *(dispRight + y * m_imgWidth + xr);

            if(abs(dl - dr) > lrcThreshold)
                *(dispLeft + y * m_imgWidth + x) = lrcInvalid;
        }
    }

    return SUCCESS;
}

stereoSGM::status_t stereoSGM::f_medianFilter3x3
(
    int8_t *src,
    int8_t *dst
)
{
    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
        {
            array<int8_t, 9> a;
            for(int wy = 0; wy < 2; wy++)
            {
                for(int wx = 0; wx < 2; wx++)
                {
                    int arrayIdx = wy * 3 + wx;
                    int sy = y + wy;
                    int sx = x + wx;
                    a[arrayIdx] = *(src + sy * m_imgWidth + sx);
                }
            }

            sort(a.begin(), a.end());
            *(dst + y * m_imgWidth + x) = a[4];
        }
    }

    return SUCCESS;
}

