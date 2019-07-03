#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include "stereoSGM.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
#include <cstring>

using namespace std;
using namespace cv;

// ------------------------------------------------- //
// census transform over a 5x5 window.               //
// it's caller's job to pad the border by 2 elements //
// a 5x5 window has 24 elements comparing with the   //
// center element so the output can be encoded in    //
// int32_t (lower 24 bits are useful).               //
// the mapping of the window elements position and   //
// output bit position (center-skipped) is like:     //
//      0    1    2    3    4                        //
//   |----+----+----+----+----|                      //
// 0 | 23 | 22 | 21 | 20 | 19 |                      // 
//   |----+----+----+----+----|                      //
// 1 | 18 | 17 | 16 | 15 | 14 |                      // 
//   |----+----+----+----+----|                      //
// 2 | 13 | 12 |  * | 11 | 10 |                      // 
//   |----+----+----+----+----|                      //
// 3 |  9 |  8 |  7 |  6 |  5 |                      // 
//   |----+----+----+----+----|                      //
// 4 |  4 |  3 |  2 |  1 |  0 |                      // 
//   |----+----+----+----+----|                      //
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
            for(int wy = 0; wy < ctWinSize; wy++)
            {
                for(int wx = 0; wx < ctWinSize; wx++)
                {
                    int32_t bitPos = ctDataMsb - (wy * ctWinSize + wx);

                    // ----------------------------- //
                    // skip the center element       //
                    // ----------------------------- //
                    if(bitPos != 11)
                    {
                        int32_t srcDataOfst = (dy + wy) * srcLinePitch + (dx + wx);
                        uint8_t srcData = *(src + srcDataOfst);
                        // -------------------------------------------- //
                        // for the bit position less then 11 (center),  //
                        // +1 to adjust the position, fit the hole of   //
                        // center                                       //
                        // -------------------------------------------- //
                        bitPos = bitPos < 11 ? bitPos + 1 : bitPos;
                        dstData |= (srcData > srcCenter) << bitPos;
                    }
                }
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

stereoSGM::status_t stereoSGM::f_aggregateCost
(
    stereoSGMCostCube<int32_t>  &matchCost,
    stereoSGMCostCube<int32_t> &sumCost
)
{
    stereoSGMCostCube<int32_t> *pathCostPtr[8];

    for(int i = 0; i < 8; i++)
        pathCostPtr[i] = new stereoSGMCostCube<int32_t>(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);

    f_getPathCost<L0>(matchCost, *(pathCostPtr[0]));
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
    for(int y = 1; y < m_imgHeight - 1; y++)
    {
        for(int x = 1; x < m_imgWidth - 1; x++)
        {
            array<int8_t, 9> a;
            for(int wy = -1; wy < 1; wy++)
            {
                for(int wx = -1; wx < 1; wx++)
                {
                    int arrayIdx = (wy + 1) * 3 + (wx + 1);
                    int sy = y + wy;
                    int sx = x + wx;
                    a[arrayIdx] = *(src + sy * m_imgWidth + sx);
                }
            }

            sort(a.begin(), a.end());
            *(dst + y * m_imgWidth + x) = a[4];
        }
    }

    for(int x = 0; x < m_imgWidth; x++)
    {
        *(dst + x) = *(src + x);
        *(dst + (m_imgHeight - 1) * m_imgWidth + x) = *(src + (m_imgHeight - 1) * m_imgWidth + x);
    }

    for(int y = 1; y < m_imgHeight - 1; y++)
    {
        *(dst + y * m_imgWidth) = *(src + y * m_imgWidth);
        *(dst + y * m_imgWidth + m_imgWidth - 1) = *(src + y * m_imgWidth + m_imgWidth - 1);
    }

    return SUCCESS;
}

stereoSGM::status_t stereoSGM::compute
(
    uint8_t *imgLeft,
    uint8_t *imgRight
)
{
    int32_t *ctLeft    = new int32_t[m_imgWidth * m_imgHeight];
    int32_t *ctRight   = new int32_t[m_imgWidth * m_imgHeight];
    int8_t  *dispLeft  = new int8_t [m_imgWidth * m_imgHeight];
    int8_t  *dispRight = new int8_t [m_imgWidth * m_imgHeight];
    stereoSGMCostCube<int32_t> matchCost(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);
    stereoSGMCostCube<int32_t> sumCost(m_imgWidth, m_imgHeight, m_imgDisp, m_imgDisp);

    f_censusTransform5x5(imgLeft,  ctLeft);
    f_censusTransform5x5(imgRight, ctRight);
    f_getMatchCost(ctLeft, ctRight, matchCost);
    f_aggregateCost(matchCost, sumCost);
    f_pickDisparity<LEFT>(sumCost, dispLeft);
    f_pickDisparity<RIGHT>(sumCost, dispRight);
    f_checkLeftRight(dispLeft, dispRight);
    f_medianFilter3x3(dispLeft, m_dispMap);

    delete []ctLeft;
    delete []ctRight;
    delete []dispLeft;
    delete []dispRight;

    return SUCCESS;
}

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
